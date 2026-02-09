"""
Train sequence + static models to predict destination port from AIS voyages.

Assumes preprocessed CSVs per year in --data-dir:

    ais_points_YYYY.csv
    voyages_YYYY.csv
    vessels_YYYY.csv
    voyage_catches_YYYY.csv   (optional, adds catch-based static features)

Supports three model types:

    --model-type tft          : TFTClassifier (your existing model)
    --model-type bilstm       : BiLSTM + static features (ModelA_BiLSTMWithStatic)
    --model-type transformer  : Transformer encoder + static features (ModelB_TransformerWithStatic)

Usage example (TFT):

    python src/model/train_ports.py --data-dir output --years 2016 2017 2018 --encoder-length 64 --hide-last-hours 3 --encoder-window-mode tail --batch-size 64 --epochs 30 --remove-statics mmsi --model-type tft

Usage example (BiLSTM baseline):

    python src/model/train_ports.py --data-dir output --years 2016 2017 2018 --encoder-length 64 --hide-last-hours 3 --encoder-window-mode tail --batch-size 64 --epochs 30 --model-type bilstm
"""

import os
import argparse
from typing import List, Dict, Tuple, Optional
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Assumes you define these three in separate files
from model import TFTClassifier
from transformer import ModelB_TransformerWithStatic
from LSTMandStatics import ModelA_BiLSTMWithStatic


# ---------- constants: same region as preprocessing ----------
MIN_LAT, MAX_LAT = 69.2, 73.0
MIN_LON, MAX_LON = 13.0, 31.5
MIN_SOG, MAX_SOG = 0.0, 30.0
MIN_COG, MAX_COG = 0.0, 360.0

AIS_STEP_MINUTES = 5  # 5-minute steps in your preprocessing

# master static feature names; we will filter these based on CLI
# continuous statics (from voyages + catches)
MASTER_STATIC_CONT_FEATURES = [
    "length_m",
    "draught_m",
    "engine_kw",
    "gross_tonnage",
    "total_catch_kg",    # from voyages_YYYY.csv (already there in your preprocessing)
    "catch_num_species", # from voyage_catches (aggregated)
    "catch_entropy",     # from voyage_catches (aggregated)
]
# categorical statics
MASTER_STATIC_CAT_FEATURES = [
    "gear",
    "mmsi",
    "season",
    "primary_species",   # from voyage_catches or voyages (top species)
]


# ---------- data loading / merging ----------

def load_year(data_dir: str, year: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = str(year)
    ais_path = os.path.join(data_dir, f"ais_points_{y}.csv")
    voyages_path = os.path.join(data_dir, f"voyages_{y}.csv")
    vessels_path = os.path.join(data_dir, f"vessels_{y}.csv")

    if not os.path.exists(ais_path):
        raise FileNotFoundError(ais_path)
    if not os.path.exists(voyages_path):
        raise FileNotFoundError(voyages_path)
    if not os.path.exists(vessels_path):
        raise FileNotFoundError(vessels_path)

    ais = pd.read_csv(ais_path, parse_dates=["ts"])
    voyages = pd.read_csv(voyages_path, parse_dates=["start_ts", "end_ts", "label_ts"])
    vessels = pd.read_csv(vessels_path)

    ais["year"] = year
    voyages["year"] = year
    vessels["year"] = year

    return ais, voyages, vessels


def load_all_years(data_dir: str, years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ais_all, voyages_all, vessels_all = [], [], []

    for y in years:
        ais, voyages, vessels = load_year(data_dir, y)
        ais_all.append(ais)
        voyages_all.append(voyages)
        vessels_all.append(vessels)

    ais_df = pd.concat(ais_all, ignore_index=True)
    voyages_df = pd.concat(voyages_all, ignore_index=True)

    vessels_df = pd.concat(vessels_all, ignore_index=True)
    vessels_df = vessels_df.sort_values("year")
    vessels_df = vessels_df.drop_duplicates(subset=["mmsi"], keep="last")

    return ais_df, voyages_df, vessels_df


def add_season_and_catch_features(
    data_dir: str,
    years: List[int],
    voyages_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    - Adds a 'season' categorical column to voyages_df based on start_ts (or label_ts).
    - Reads voyage_catches_YYYY.csv for the requested years (if present) and aggregates:
        * catch_num_species   (per voyage)
        * catch_entropy       (Shannon entropy over species composition)
        * primary_species     (species with max total_rundvekt_kg; FAO name where possible)
    - Merges these catch-based features into voyages_df on voyage_id.
    Returns (voyages_df_enriched, catches_df_all_or_None).
    """

    voyages_df = voyages_df.copy()

    # ---- season from start_ts (fallback to label_ts) ----
    def _season_from_ts(ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return "Unknown"
        m = ts.month
        # Your spec:
        # • Winter: Jan – March, plus April
        # • Spring: May – June
        # • Summer: July – August
        # • Autumn: The rest
        if m in (1, 2, 3, 4):
            return "Winter"
        elif m in (5, 6):
            return "Spring"
        elif m in (7, 8):
            return "Summer"
        else:
            return "Autumn"

    if "start_ts" in voyages_df.columns:
        ts_col = voyages_df["start_ts"].where(voyages_df["start_ts"].notna(), voyages_df.get("label_ts"))
    else:
        ts_col = voyages_df.get("label_ts")

    voyages_df["season"] = ts_col.apply(_season_from_ts)

    # ---- aggregate voyage_catches_YYYY.csv ----
    all_catches = []
    for y in years:
        y_str = str(y)
        path = os.path.join(data_dir, f"voyage_catches_{y_str}.csv")
        if os.path.exists(path):
            print(f"[INFO] Loading catches from {path}")
            c = pd.read_csv(path)
            all_catches.append(c)
        else:
            print(f"[INFO] No voyage_catches file found for year {y_str} (path={path})")

    if not all_catches:
        print("[INFO] No voyage_catches_YYYY.csv files found – catch-based features will be zero / Unknown.")
        voyages_df["catch_num_species"] = np.nan
        voyages_df["catch_entropy"] = np.nan
        # keep any existing primary_species, otherwise fill later
        return voyages_df, None

    catches_df = pd.concat(all_catches, ignore_index=True)

    # ensure columns exist
    if "voyage_id" not in catches_df.columns or "total_rundvekt_kg" not in catches_df.columns:
        print("[WARN] voyage_catches files missing 'voyage_id' or 'total_rundvekt_kg'; skipping catch-based features.")
        voyages_df["catch_num_species"] = np.nan
        voyages_df["catch_entropy"] = np.nan
        return voyages_df, catches_df

    catches_df["total_rundvekt_kg"] = pd.to_numeric(
        catches_df["total_rundvekt_kg"], errors="coerce"
    ).fillna(0.0)

    # num species per voyage (using FAO code)
    num_species = (
        catches_df.groupby("voyage_id")["art_fao_code"]
        .nunique()
        .reset_index(name="catch_num_species")
    )

    # entropy per voyage
    def _shannon_entropy(group: pd.DataFrame) -> float:
        w = group["total_rundvekt_kg"].to_numpy(dtype=float)
        tot = w.sum()
        if tot <= 0.0:
            return 0.0
        p = w / tot
        return float(-np.sum(p * np.log(p + 1e-12)))

    entropy = (
        catches_df.groupby("voyage_id")
        .apply(_shannon_entropy)
        .reset_index(name="catch_entropy")
    )

    # primary species per voyage (by total weight)
    # prefer Art FAO (name), fall back to FAO code or FDIR
    def _primary_species_per_group(group: pd.DataFrame) -> str:
        # species-level aggregation
        agg = (
            group.groupby(["art_fao_code", "art_fao", "art_fdir_code", "art_fdir"], dropna=False)["total_rundvekt_kg"]
            .sum()
            .reset_index()
        )
        if agg.empty:
            return "Unknown"
        top = agg.sort_values("total_rundvekt_kg", ascending=False).iloc[0]
        for col in ["art_fao", "art_fdir", "art_fao_code", "art_fdir_code"]:
            val = top.get(col)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return "Unknown"

    primary_species = (
        catches_df.groupby("voyage_id")
        .apply(_primary_species_per_group)
        .reset_index(name="primary_species_from_catches")
    )

    # merge all on voyage_id
    voyages_df = voyages_df.merge(num_species, on="voyage_id", how="left")
    voyages_df = voyages_df.merge(entropy, on="voyage_id", how="left")
    voyages_df = voyages_df.merge(primary_species, on="voyage_id", how="left")

    # if voyages already has "primary_species", keep it; else use from catches
    if "primary_species" not in voyages_df.columns:
        voyages_df["primary_species"] = voyages_df["primary_species_from_catches"]
    else:
        # fill missing primary_species from catches where available
        mask = voyages_df["primary_species"].isna() | (voyages_df["primary_species"].astype(str).str.len() == 0)
        voyages_df.loc[mask, "primary_species"] = voyages_df.loc[mask, "primary_species_from_catches"]

    voyages_df.drop(columns=[c for c in ["primary_species_from_catches"] if c in voyages_df.columns], inplace=True)

    return voyages_df, catches_df


# ---------- dataset ----------

class VoyagePortDataset(Dataset):
    """
    One sample = one voyage.

    Encoder inputs:
        - AIS sequence: lat/lon/sog/cog normalized, length = encoder_length,
          taken from the **observed part** of the track (last N hours can be hidden).

    Static inputs:
        - continuous: subset of MASTER_STATIC_CONT_FEATURES
        - categorical: subset of MASTER_STATIC_CAT_FEATURES

    Target:
        - destination port class (0 .. num_ports-1)
    """

    def __init__(
        self,
        ais_df: pd.DataFrame,
        voyages_df: pd.DataFrame,
        vessels_df: pd.DataFrame,
        encoder_length: int,
        port_id_to_idx: Dict[int, int],
        gear_to_idx: Dict[str, int],
        mmsi_to_idx: Dict[str, int],
        season_to_idx: Dict[str, int],
        species_to_idx: Dict[str, int],
        static_cont_features: List[str],
        static_cat_features: List[str],
        hide_last_n_points: int = 0,
        voyage_ids: Optional[List[str]] = None,
        window_mode: str = "tail",
        name: str = "",
    ):
        """
        window_mode:
            - "tail": use LAST encoder_length points of the observed part
                      (padding at BEGINNING) – default.
            - "head": use FIRST encoder_length points of the observed part
                      (padding at END).
        """
        super().__init__()
        assert window_mode in ("tail", "head"), "window_mode must be 'tail' or 'head'"

        self.encoder_length = encoder_length
        self.port_id_to_idx = port_id_to_idx
        self.gear_to_idx = gear_to_idx
        self.mmsi_to_idx = mmsi_to_idx
        self.season_to_idx = season_to_idx
        self.species_to_idx = species_to_idx
        self.static_cont_features = static_cont_features
        self.static_cat_features = static_cat_features
        self.hide_last_n_points = hide_last_n_points
        self.window_mode = window_mode
        self.name = name

        # index ais by voyage_id for fast lookup
        self.ais_grouped = ais_df.sort_values("ts").groupby("voyage_id")

        # merge voyages with vessels to get static info
        merged = voyages_df.merge(
            vessels_df,
            on="mmsi",
            how="left",
            suffixes=("", "_vessel"),
        )

        if voyage_ids is not None:
            merged = merged[merged["voyage_id"].isin(voyage_ids)]

        # keep only voyages with known label_port_id that exist in AIS group and port mapping
        valid_rows = []
        for _, row in merged.iterrows():
            vid = row["voyage_id"]
            port_id = row["label_port_id"]
            if pd.isna(vid) or pd.isna(port_id):
                continue
            if vid not in self.ais_grouped.groups:
                continue
            if int(port_id) not in port_id_to_idx:
                continue

            # Also require that after hiding future points we still have at least 1 point
            df_traj_full = self.ais_grouped.get_group(vid)
            if self.hide_last_n_points > 0 and len(df_traj_full) > self.hide_last_n_points:
                df_obs = df_traj_full.iloc[:-self.hide_last_n_points]
            else:
                df_obs = df_traj_full

            if len(df_obs) == 0:
                continue

            valid_rows.append(row)

        self.voyages = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(
            f"[Dataset] Using {len(self.voyages)} voyages from {len(merged)} raw voyages "
            f"({self.name}, window_mode={self.window_mode})"
        )

    def __len__(self):
        return len(self.voyages)

    def _normalize_track(self, df_traj: pd.DataFrame):
        """
        df_traj: rows for one voyage, sorted by ts, with columns lat, lon, sog, cog.

        window_mode == "tail":
            - use LAST encoder_length points from the observed part
            - pad at the BEGINNING if the sequence is shorter.

        window_mode == "head":
            - use FIRST encoder_length points from the observed part
            - pad at the END if shorter.
        """
        if self.window_mode == "tail":
            # use LAST encoder_length points
            df_traj = df_traj.tail(self.encoder_length)

            # pad at the BEGINNING if shorter
            pad_len = self.encoder_length - len(df_traj)
            lat = df_traj["lat"].to_numpy()
            lon = df_traj["lon"].to_numpy()
            sog = df_traj["sog"].to_numpy()
            cog = (df_traj["cog"].to_numpy() % 360.0)

            if pad_len > 0:
                lat = np.concatenate([np.full(pad_len, MIN_LAT, dtype=float), lat])
                lon = np.concatenate([np.full(pad_len, MIN_LON, dtype=float), lon])
                sog = np.concatenate([np.zeros(pad_len, dtype=float), sog])
                cog = np.concatenate([np.zeros(pad_len, dtype=float), cog])

        else:  # "head"
            # use FIRST encoder_length points
            df_traj = df_traj.head(self.encoder_length)

            # pad at the END if shorter
            pad_len = self.encoder_length - len(df_traj)
            lat = df_traj["lat"].to_numpy()
            lon = df_traj["lon"].to_numpy()
            sog = df_traj["sog"].to_numpy()
            cog = (df_traj["cog"].to_numpy() % 360.0)

            if pad_len > 0:
                lat = np.concatenate([lat, np.full(pad_len, MIN_LAT, dtype=float)])
                lon = np.concatenate([lon, np.full(pad_len, MIN_LON, dtype=float)])
                sog = np.concatenate([sog, np.zeros(pad_len, dtype=float)])
                cog = np.concatenate([cog, np.zeros(pad_len, dtype=float)])

        # normalize to ~[0,1]
        lat_n = (lat - MIN_LAT) / (MAX_LAT - MIN_LAT)
        lon_n = (lon - MIN_LON) / (MAX_LON - MIN_LON)
        sog_n = (sog - MIN_SOG) / (MAX_SOG - MIN_SOG)
        cog_n = (cog - MIN_COG) / (MAX_COG - MIN_COG)

        lat_t = torch.tensor(lat_n, dtype=torch.float32).unsqueeze(-1)  # (T,1)
        lon_t = torch.tensor(lon_n, dtype=torch.float32).unsqueeze(-1)
        sog_t = torch.tensor(sog_n, dtype=torch.float32).unsqueeze(-1)
        cog_t = torch.tensor(cog_n, dtype=torch.float32).unsqueeze(-1)

        enc_len = torch.tensor(self.encoder_length, dtype=torch.long)

        return {
            "lat": lat_t,
            "lon": lon_t,
            "sog": sog_t,
            "cog": cog_t,
        }, enc_len

    def __getitem__(self, idx):
        row = self.voyages.iloc[idx]
        vid = row["voyage_id"]
        port_id = int(row["label_port_id"])
        port_class = self.port_id_to_idx[port_id]

        # Full AIS track
        df_traj_full = self.ais_grouped.get_group(vid).sort_values("ts")

        # --- hide the last N points to simulate missing future ---
        if self.hide_last_n_points > 0 and len(df_traj_full) > self.hide_last_n_points:
            df_traj_obs = df_traj_full.iloc[:-self.hide_last_n_points]
        else:
            df_traj_obs = df_traj_full

        enc_vars, enc_len = self._normalize_track(df_traj_obs)

        # Decoder input: single dummy time step (only used by TFT)
        T_dec = 1
        dec_vars = {
            "dec_time": torch.zeros(T_dec, 1, dtype=torch.float32)
        }

        # Static continuous
        static_cont_vals = []
        for c in self.static_cont_features:
            val = row.get(c, np.nan)
            static_cont_vals.append(0.0 if pd.isna(val) else float(val))
        static_cont = torch.tensor(static_cont_vals, dtype=torch.float32)  # (n_cont,)

        # Static categorical
        static_cat_vals: List[int] = []
        for feat in self.static_cat_features:
            if feat == "gear":
                gear = str(row.get("gear_primary", "Unknown"))
                gear_idx = self.gear_to_idx.get(gear, self.gear_to_idx["Unknown"])
                static_cat_vals.append(gear_idx)
            elif feat == "mmsi":
                mmsi_str = str(row.get("mmsi", "Unknown"))
                mmsi_idx = self.mmsi_to_idx.get(mmsi_str, self.mmsi_to_idx["Unknown"])
                static_cat_vals.append(mmsi_idx)
            elif feat == "season":
                season_str = str(row.get("season", "Unknown"))
                season_idx = self.season_to_idx.get(season_str, self.season_to_idx["Unknown"])
                static_cat_vals.append(season_idx)
            elif feat == "primary_species":
                species_str = str(row.get("primary_species", "Unknown"))
                species_idx = self.species_to_idx.get(species_str, self.species_to_idx["Unknown"])
                static_cat_vals.append(species_idx)
            else:
                raise ValueError(f"Unknown static categorical feature: {feat}")

        static_cat = torch.tensor(static_cat_vals, dtype=torch.long)  # (n_cat,)

        target = torch.tensor(port_class, dtype=torch.long)
        eta_target = torch.tensor(0.0, dtype=torch.float32)  # unused but kept for API

        return enc_vars, dec_vars, enc_len, static_cont, static_cat, target, eta_target


def collate_batch(batch):
    enc_vars_batch: Dict[str, torch.Tensor] = {}
    dec_vars_batch: Dict[str, torch.Tensor] = {}
    enc_lens = []
    static_conts = []
    static_cats = []
    targets = []
    eta_targets = []

    enc_keys = list(batch[0][0].keys())
    dec_keys = list(batch[0][1].keys())

    for enc_vars, dec_vars, enc_len, static_cont, static_cat, target, eta_t in batch:
        for k in enc_keys:
            enc_vars_batch.setdefault(k, []).append(enc_vars[k])
        for k in dec_keys:
            dec_vars_batch.setdefault(k, []).append(dec_vars[k])
        enc_lens.append(enc_len)
        static_conts.append(static_cont)
        static_cats.append(static_cat)
        targets.append(target)
        eta_targets.append(eta_t)

    for k in enc_keys:
        enc_vars_batch[k] = torch.stack(enc_vars_batch[k], dim=0)
    for k in dec_keys:
        dec_vars_batch[k] = torch.stack(dec_vars_batch[k], dim=0)

    enc_lens = torch.stack(enc_lens, dim=0)
    static_conts = torch.stack(static_conts, dim=0)
    static_cats = torch.stack(static_cats, dim=0)
    targets = torch.stack(targets, dim=0)
    eta_targets = torch.stack(eta_targets, dim=0)

    return enc_vars_batch, dec_vars_batch, enc_lens, static_conts, static_cats, targets, eta_targets


# ---------- evaluation helper ----------

def evaluate_model(model, loader, device, model_type: str):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_cnt = 0

    with torch.no_grad():
        for batch in loader:
            enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, eta_targets = batch

            enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
            dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
            enc_lens = enc_lens.to(device)
            static_cont = static_cont.to(device)
            static_cat = static_cat.to(device)
            targets = targets.to(device)

            if model_type == "tft":
                (logits, eta_q), extras = model(
                    enc_vars=enc_vars,
                    dec_vars=dec_vars,
                    enc_lengths=enc_lens,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )
                logits = logits.squeeze(1)  # (B, num_ports)
            else:
                # Build (B, T, 4) from encoder vars
                x_seq = torch.cat(
                    [enc_vars["lat"], enc_vars["lon"], enc_vars["sog"], enc_vars["cog"]],
                    dim=-1,
                )  # (B, T, 4)
                logits = model(
                    x_seq=x_seq,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )  # (B, num_ports)

            loss = F.cross_entropy(logits, targets)

            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_cnt += targets.size(0)

    avg_loss = total_loss / max(total_cnt, 1)
    acc = total_correct / max(total_cnt, 1)
    return avg_loss, acc


def inspect_one_test_sample(
    model,
    test_loader,
    device,
    idx_to_port_id,
    idx_to_gear,
    idx_to_mmsi,
    idx_to_season,
    idx_to_species,
    static_cat_features: List[str],
    enc_feature_names=("lat", "lon", "sog", "cog"),
    top_k=5,
    model_type: str = "tft",
):
    """
    Take one batch from test_loader, inspect the first sample:
      - print some of the encoder inputs
      - print static features (continuous vector + decoded categorical)
      - print top-k predicted ports with probabilities

    Works for all model types; TFT-specific extras are ignored here.
    """
    model.eval()
    batch = next(iter(test_loader))

    enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, eta_targets = batch

    # Move to device
    enc_vars_dev = {k: v.to(device) for k, v in enc_vars.items()}
    dec_vars_dev = {k: v.to(device) for k, v in dec_vars.items()}
    enc_lens_dev = enc_lens.to(device)
    static_cont_dev = static_cont.to(device)
    static_cat_dev = static_cat.to(device)
    targets_dev = targets.to(device)

    with torch.no_grad():
        if model_type == "tft":
            (logits, eta_q), extras = model(
                enc_vars=enc_vars_dev,
                dec_vars=dec_vars_dev,
                enc_lengths=enc_lens_dev,
                static_cont=static_cont_dev,
                static_cat=static_cat_dev,
            )
            logits = logits.squeeze(1)
        else:
            x_seq = torch.cat(
                [enc_vars_dev["lat"], enc_vars_dev["lon"], enc_vars_dev["sog"], enc_vars_dev["cog"]],
                dim=-1,
            )
            logits = model(
                x_seq=x_seq,
                static_cont=static_cont_dev,
                static_cat=static_cat_dev,
            )

        probs = torch.softmax(logits, dim=-1)  # (B, num_ports)

    # Inspect only first sample in this batch
    b = 0

    sample_probs = probs[b].cpu().numpy()
    true_idx = targets_dev[b].item()
    true_port_id = idx_to_port_id[true_idx]

    print("\n=== Inspecting one test sample ===")
    print(f"Batch size: {probs.size(0)}")
    print(f"True class index: {true_idx}  -> port_id={true_port_id}")

    # ----- show some encoder inputs -----
    print("\nEncoder inputs (last 5 timesteps):")
    for feat in enc_feature_names:
        series = enc_vars[feat][b, :, 0].cpu().numpy()  # (T,)
        print(f"  {feat}: {series[-5:]}")  # last 5 values

    # ----- static continuous features -----
    sc = static_cont[b].cpu().numpy().tolist()
    print("\nStatic continuous (raw vector):")
    for i, v in enumerate(sc):
        print(f"  cont[{i}]: {v:.3f}")

    # ----- static categorical -----
    print("\nStatic categorical (decoded):")
    for j, feat in enumerate(static_cat_features):
        idx = static_cat[b, j].item()
        if feat == "gear":
            label = idx_to_gear.get(idx, f"<gear-idx-{idx}>")
        elif feat == "mmsi":
            label = idx_to_mmsi.get(idx, f"<mmsi-idx-{idx}>")
        elif feat == "season":
            label = idx_to_season.get(idx, f"<season-idx-{idx}>")
        elif feat == "primary_species":
            label = idx_to_species.get(idx, f"<species-idx-{idx}>")
        else:
            label = f"<{feat}-idx-{idx}>"
        print(f"  {feat}: {label} (idx={idx})")

    # ----- top-k predicted ports -----
    top_k = min(top_k, sample_probs.shape[0])
    topk_idx = sample_probs.argsort()[::-1][:top_k]

    print(f"\nTop-{top_k} predicted ports (by probability):")
    for rank, cls_idx in enumerate(topk_idx, start=1):
        port_id = idx_to_port_id[cls_idx]
        p = sample_probs[cls_idx]
        marker = " <-- TRUE" if cls_idx == true_idx else ""
        print(f"  #{rank}: class_idx={cls_idx:3d}  port_id={port_id:5d}  p={p:.3f}{marker}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Train models to predict destination port with partial tracks")
    parser.add_argument("--data-dir", type=str, default="output")
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018])
    parser.add_argument("--encoder-length", type=int, default=64)
    parser.add_argument(
        "--hide-last-hours",
        type=float,
        default=0.0,
        help="Hide this many hours at the end of each voyage before building the encoder window",
    )
    parser.add_argument(
        "--encoder-window-mode",
        type=str,
        choices=["tail", "head"],
        default="tail",
        help=(
            "'tail' (default): use last encoder_length points of observed track "
            "(padding at beginning). "
            "'head': use first encoder_length points of observed track "
            "(padding at end)."
        ),
    )
    parser.add_argument(
        "--remove-statics",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Static features to drop from training/testing. "
            "Allowed: "
            "length_m, draught_m, engine_kw, gross_tonnage, total_catch_kg, "
            "catch_num_species, catch_entropy, "
            "gear, mmsi, season, primary_species"
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tft", "bilstm", "transformer"],
        default="tft",
        help="Which model architecture to train.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep-hide-hours",
        action="store_true",
        help="After training, evaluate test accuracy for multiple hide_last_hours values and plot accuracy vs hours.",
    )
    parser.add_argument("--sweep-hide-start", type=float, default=3.0)
    parser.add_argument("--sweep-hide-end", type=float, default=5.0)
    parser.add_argument("--sweep-hide-step", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model type: {args.model_type}")

    print(f"[INFO] Loading data from {args.data_dir} for years {args.years}")
    ais_df, voyages_df, vessels_df = load_all_years(args.data_dir, args.years)

    # ---- add season + catch-based static features from voyage_catches ----
    voyages_df, catches_df = add_season_and_catch_features(args.data_dir, args.years, voyages_df)

    # ---- port / gear / mmsi / season / species mappings ----
    unique_port_ids = sorted(voyages_df["label_port_id"].dropna().astype(int).unique().tolist())
    port_id_to_idx = {pid: i for i, pid in enumerate(unique_port_ids)}
    num_ports = len(port_id_to_idx)
    print(f"[INFO] Number of destination ports (classes): {num_ports}")

    gear_values = voyages_df["gear_primary"].astype(str).fillna("Unknown")
    gear_values = list(sorted(set(gear_values.tolist() + ["Unknown"])))
    gear_to_idx = {g: i for i, g in enumerate(gear_values)}
    print(f"[INFO] Number of gear types: {len(gear_values)}")

    mmsi_values = voyages_df["mmsi"].astype(str).fillna("Unknown")
    mmsi_values = list(sorted(set(mmsi_values.tolist() + ["Unknown"])))
    mmsi_to_idx = {m: i for i, m in enumerate(mmsi_values)}
    print(f"[INFO] Number of MMSI categories: {len(mmsi_values)}")

    season_values = voyages_df["season"].astype(str).fillna("Unknown")
    season_values = list(sorted(set(season_values.tolist() + ["Unknown"])))
    season_to_idx = {s: i for i, s in enumerate(season_values)}
    print(f"[INFO] Number of seasons: {len(season_values)}")

    species_col = voyages_df.get("primary_species")
    if species_col is None:
        species_values = ["Unknown"]
    else:
        species_values = species_col.astype(str).fillna("Unknown").tolist()
        species_values = list(sorted(set(species_values + ["Unknown"])))
    species_to_idx = {s: i for i, s in enumerate(species_values)}
    print(f"[INFO] Number of primary_species categories: {len(species_values)}")

    idx_to_port_id = {idx: pid for pid, idx in port_id_to_idx.items()}
    idx_to_gear = {idx: g for g, idx in gear_to_idx.items()}
    idx_to_mmsi = {idx: m for m, idx in mmsi_to_idx.items()}
    idx_to_season = {idx: s for s, idx in season_to_idx.items()}
    idx_to_species = {idx: s for s, idx in species_to_idx.items()}

    # ---- choose static features based on CLI ----
    remove_set = set(args.remove_statics or [])

    static_cont_features = [f for f in MASTER_STATIC_CONT_FEATURES if f not in remove_set]
    static_cat_features = [f for f in MASTER_STATIC_CAT_FEATURES if f not in remove_set]

    print(f"[INFO] Using static continuous features: {static_cont_features}")
    print(f"[INFO] Using static categorical features: {static_cat_features}")

    # ---- split voyage IDs into train/val/test ----
    all_voyage_ids = voyages_df["voyage_id"].dropna().unique().tolist()
    rng = np.random.default_rng(seed=args.seed)
    rng.shuffle(all_voyage_ids)

    n_total = len(all_voyage_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ids = all_voyage_ids[:n_train]
    val_ids = all_voyage_ids[n_train:n_train + n_val]
    test_ids = all_voyage_ids[n_train + n_val:]

    print(f"[INFO] Voyage split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ---- compute how many points to hide ----
    if args.hide_last_hours > 0:
        points_per_hour = int(60 / AIS_STEP_MINUTES)  # 12 for 5-min data
        hide_last_n_points = int(args.hide_last_hours * points_per_hour)
    else:
        hide_last_n_points = 0

    print(
        f"[INFO] Hiding last {args.hide_last_hours} hours "
        f"(≈ {hide_last_n_points} AIS points) of each voyage before feeding to encoder.",
    )
    print(f"[INFO] Encoder window mode: {args.encoder_window_mode}")

    # ---- datasets ----
    dataset_kwargs = dict(
        ais_df=ais_df,
        voyages_df=voyages_df,
        vessels_df=vessels_df,
        encoder_length=args.encoder_length,
        port_id_to_idx=port_id_to_idx,
        gear_to_idx=gear_to_idx,
        mmsi_to_idx=mmsi_to_idx,
        season_to_idx=season_to_idx,
        species_to_idx=species_to_idx,
        static_cont_features=static_cont_features,
        static_cat_features=static_cat_features,
        hide_last_n_points=hide_last_n_points,
        window_mode=args.encoder_window_mode,
    )

    train_dataset = VoyagePortDataset(
        **dataset_kwargs,
        voyage_ids=train_ids,
        name="train",
    )
    val_dataset = VoyagePortDataset(
        **dataset_kwargs,
        voyage_ids=val_ids,
        name="val",
    )
    test_dataset = VoyagePortDataset(
        **dataset_kwargs,
        voyage_ids=test_ids,
        name="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # ---- model ----
    enc_var_dims = {"lat": 1, "lon": 1, "sog": 1, "cog": 1}
    dec_var_dims = {"dec_time": 1}

    # build cat cardinalities in same order as static_cat_features
    static_cat_cardinalities: List[int] = []
    for f in static_cat_features:
        if f == "gear":
            static_cat_cardinalities.append(len(gear_to_idx))
        elif f == "mmsi":
            static_cat_cardinalities.append(len(mmsi_to_idx))
        elif f == "season":
            static_cat_cardinalities.append(len(season_to_idx))
        elif f == "primary_species":
            static_cat_cardinalities.append(len(species_to_idx))
        else:
            raise ValueError(f"Unknown categorical static feature: {f}")

    if args.model_type == "tft":
        model = TFTClassifier(
            enc_var_dims=enc_var_dims,
            dec_var_dims=dec_var_dims,
            static_cont_dim=len(static_cont_features),
            static_cat_cardinalities=static_cat_cardinalities,
            static_cat_emb_dim=16,
            d_model=64,
            lstm_layers=1,
            dropout=0.1,
            num_heads=4,
            num_classes=num_ports,
            pred_len=1,
            eta_quantiles=(),
        ).to(device)
    elif args.model_type == "bilstm":
        model = ModelA_BiLSTMWithStatic(
            ts_input_dim=4,  # lat/lon/sog/cog
            static_cont_dim=len(static_cont_features),
            static_cat_cardinalities=static_cat_cardinalities,
            num_classes=num_ports,
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
            static_emb_dim=16,
        ).to(device)
    elif args.model_type == "transformer":
        model = ModelB_TransformerWithStatic(
            ts_input_dim=4,
            static_cont_dim=len(static_cont_features),
            static_cat_cardinalities=static_cat_cardinalities,
            num_classes=num_ports,
            d_model=64,
            static_emb_dim=16,
            n_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- training loop with val + best model tracking ----
    best_val_acc = 0.0
    best_state = None

    print("[TRAIN] Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_cnt = 0

        for batch in train_loader:
            enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, eta_targets = batch

            enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
            dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
            enc_lens = enc_lens.to(device)
            static_cont = static_cont.to(device)
            static_cat = static_cat.to(device)
            targets = targets.to(device)

            if args.model_type == "tft":
                (logits, eta_q), extras = model(
                    enc_vars=enc_vars,
                    dec_vars=dec_vars,
                    enc_lengths=enc_lens,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )
                logits = logits.squeeze(1)
            else:
                x_seq = torch.cat(
                    [enc_vars["lat"], enc_vars["lon"], enc_vars["sog"], enc_vars["cog"]],
                    dim=-1,
                )
                logits = model(
                    x_seq=x_seq,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )

            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_cnt += targets.size(0)

        train_loss = total_loss / max(total_cnt, 1)
        train_acc = total_correct / max(total_cnt, 1)

        val_loss, val_acc = evaluate_model(model, val_loader, device, args.model_type)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("[TRAIN] Training finished.")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Loaded best model (val_acc={best_val_acc:.3f}) for final test evaluation.")

    test_loss, test_acc = evaluate_model(model, test_loader, device, args.model_type)
    print(f"[TEST] loss={test_loss:.4f}  acc={test_acc:.3f}")

    # ---------- feature importance (only for TFT) ----------
    if args.model_type == "tft":
        model.eval()
        enc_alpha_sum = None
        n_enc_batches = 0
        enc_feature_names = list(enc_var_dims.keys())  # ["lat","lon","sog","cog"]

        with torch.no_grad():
            for batch in test_loader:
                enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, eta_targets = batch
                enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
                dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
                enc_lens = enc_lens.to(device)
                static_cont = static_cont.to(device)
                static_cat = static_cat.to(device)

                (_, _), extras = model(
                    enc_vars=enc_vars,
                    dec_vars=dec_vars,
                    enc_lengths=enc_lens,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )

                enc_alpha = extras.get("enc_alpha", None)
                if enc_alpha is None:
                    continue
                # enc_alpha: (B, T_enc, V_enc) -> mean over batch and time
                batch_mean = enc_alpha.mean(dim=(0, 1))  # (V_enc,)
                if enc_alpha_sum is None:
                    enc_alpha_sum = batch_mean
                else:
                    enc_alpha_sum += batch_mean
                n_enc_batches += 1

        if enc_alpha_sum is not None and n_enc_batches > 0:
            enc_alpha_mean = enc_alpha_sum / n_enc_batches
            print("\n[Feature importance] Encoder variables (test-set average):")
            for name, score in zip(enc_feature_names, enc_alpha_mean.cpu().numpy()):
                print(f"   {name}: {score:.4f}")
        else:
            print("\n[Feature importance] Encoder variables: enc_alpha not found in extras (check TFTClassifier).")

        # ---------- feature importance on STATIC variables ----------
        static_alpha_sum = None
        n_static_batches = 0
        ordered_static_names = static_cont_features + static_cat_features

        with torch.no_grad():
            for batch in test_loader:
                enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, eta_targets = batch
                enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
                dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
                enc_lens = enc_lens.to(device)
                static_cont = static_cont.to(device)
                static_cat = static_cat.to(device)

                (_, _), extras = model(
                    enc_vars=enc_vars,
                    dec_vars=dec_vars,
                    enc_lengths=enc_lens,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )

                static_alpha = extras.get("static_alpha", None)
                if static_alpha is None:
                    continue
                # static_alpha: (B, V_static) -> mean over batch
                batch_mean = static_alpha.mean(dim=0)  # (V_static,)
                if static_alpha_sum is None:
                    static_alpha_sum = batch_mean
                else:
                    static_alpha_sum += batch_mean
                n_static_batches += 1

        if static_alpha_sum is not None and n_static_batches > 0 and len(ordered_static_names) > 0:
            static_alpha_mean = static_alpha_sum / n_static_batches
            print("\n[Feature importance] STATIC variables (test-set average):")
            for name, score in zip(ordered_static_names, static_alpha_mean.cpu().numpy()):
                print(f"   {name}: {score:.4f}")
        else:
            print("\n[Feature importance] STATIC variables: either none in use or static_alpha not found in extras.")
    else:
        print("\n[Feature importance] Skipped: only implemented for TFT (model_type='tft').")

    # ---------- inspect one sample ----------
    inspect_one_test_sample(
        model=model,
        test_loader=test_loader,
        device=device,
        idx_to_port_id=idx_to_port_id,
        idx_to_gear=idx_to_gear,
        idx_to_mmsi=idx_to_mmsi,
        idx_to_season=idx_to_season,
        idx_to_species=idx_to_species,
        static_cat_features=static_cat_features,
        enc_feature_names=list(enc_var_dims.keys()),
        top_k=5,
        model_type=args.model_type,
    )

    # def measure_inference_time(
    #     model,
    #     test_loader,
    #     device,
    #     model_type: str,
    #     max_samples: int = 100,
    # ):
    #     model.eval()
    #     times_ms = []

    #     seen = 0
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, _ = batch

    #             enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
    #             dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
    #             enc_lens = enc_lens.to(device)
    #             static_cont = static_cont.to(device)
    #             static_cat = static_cat.to(device)

    #             B = enc_lens.size(0)

    #             for i in range(B):
    #                 if seen >= max_samples:
    #                     return times_ms

    #                 # single-sample slicing
    #                 enc_vars_i = {k: v[i:i+1] for k, v in enc_vars.items()}
    #                 dec_vars_i = {k: v[i:i+1] for k, v in dec_vars.items()}
    #                 enc_lens_i = enc_lens[i:i+1]
    #                 static_cont_i = static_cont[i:i+1]
    #                 static_cat_i = static_cat[i:i+1]

    #                 start = time.perf_counter()

    #                 if model_type == "tft":
    #                     (logits, _), _ = model(
    #                         enc_vars=enc_vars_i,
    #                         dec_vars=dec_vars_i,
    #                         enc_lengths=enc_lens_i,
    #                         static_cont=static_cont_i,
    #                         static_cat=static_cat_i,
    #                     )
    #                 else:
    #                     x_seq = torch.cat(
    #                         [
    #                             enc_vars_i["lat"],
    #                             enc_vars_i["lon"],
    #                             enc_vars_i["sog"],
    #                             enc_vars_i["cog"],
    #                         ],
    #                         dim=-1,
    #                     )
    #                     logits = model(
    #                         x_seq=x_seq,
    #                         static_cont=static_cont_i,
    #                         static_cat=static_cat_i,
    #                     )

    #                 end = time.perf_counter()

    #                 times_ms.append((end - start) * 1000.0)
    #                 seen += 1

    #     return times_ms


    # times_ms = measure_inference_time(
    #     model=model,
    #     test_loader=test_loader,
    #     device=device,
    #     model_type=args.model_type,
    #     max_samples=100,
    # )

    # if len(times_ms) > 0:
    #     avg_time = sum(times_ms) / len(times_ms)
    #     print(f"\n[Inference timing] Average inference time per sample: {avg_time:.3f} ms")

    #     plt.figure(figsize=(8, 4))
    #     plt.plot(times_ms, marker="o", linestyle="-", alpha=0.7)
    #     plt.xlabel("Sample index")
    #     plt.ylabel("Inference time (ms)")
    #     plt.title(f"Inference time per sample ({args.model_type}, n={len(times_ms)})")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("[Inference timing] No samples measured.")

    
        # ---------- sweep inference accuracy vs hidden-last-hours ----------
    if args.sweep_hide_hours:
        import time
        import matplotlib.pyplot as plt

        print("\n[SWEEP] Evaluating accuracy vs hide_last_hours on TEST set...")

        sweep_hours = np.arange(
            args.sweep_hide_start,
            args.sweep_hide_end + 1e-9,
            args.sweep_hide_step,
            dtype=float,
        )

        sweep_acc = []
        sweep_loss = []

        for h in sweep_hours:
            # convert hours -> number of AIS points to hide
            points_per_hour = int(60 / AIS_STEP_MINUTES)  # 12 for 5-min data
            hide_last_n_points_sweep = int(round(h * points_per_hour))

            # build a fresh TEST dataset with same IDs, same mappings, different hide length
            test_dataset_sweep = VoyagePortDataset(
                ais_df=ais_df,
                voyages_df=voyages_df,
                vessels_df=vessels_df,
                encoder_length=args.encoder_length,
                port_id_to_idx=port_id_to_idx,
                gear_to_idx=gear_to_idx,
                mmsi_to_idx=mmsi_to_idx,
                season_to_idx=season_to_idx,
                species_to_idx=species_to_idx,
                static_cont_features=static_cont_features,
                static_cat_features=static_cat_features,
                hide_last_n_points=hide_last_n_points_sweep,
                voyage_ids=test_ids,
                window_mode=args.encoder_window_mode,
                name=f"test_sweep_hide_{h:.1f}h",
            )

            test_loader_sweep = DataLoader(
                test_dataset_sweep,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_batch,
            )

            # evaluate
            t0 = time.perf_counter()
            loss_h, acc_h = evaluate_model(model, test_loader_sweep, device, args.model_type)
            dt = (time.perf_counter() - t0)

            sweep_loss.append(loss_h)
            sweep_acc.append(acc_h)

            print(f"  hide_last_hours={h:.1f} -> acc={acc_h:.4f}, loss={loss_h:.4f} (eval_time={dt:.2f}s)")

        # plot accuracy vs hours
        plt.figure()
        plt.plot(sweep_hours, sweep_acc, marker="o")
        plt.xlabel("Hidden last hours (hours)")
        plt.ylabel("Test accuracy")
        plt.title(f"Accuracy vs hidden-last-hours (model={args.model_type})")
        plt.grid(True)

        out_path = os.path.join(args.data_dir, f"sweep_acc_vs_hidehours3_{args.model_type}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[SWEEP] Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
