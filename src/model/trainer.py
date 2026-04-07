"""
Train sequence + static models to predict destination port from AIS voyages.

Assumes preprocessed CSVs per year in --data-dir:

    ais_points_YYYY.csv
    voyages_YYYY.csv
    vessels_YYYY.csv
    voyage_catches_YYYY.csv   (optional, adds catch-based static features)

Supports three model types:

    --model-type tft          : TFTClassifier
    --model-type bilstm       : BiLSTM + static features (ModelA_BiLSTMWithStatic)
    --model-type transformer  : Transformer encoder + static features (ModelB_TransformerWithStatic)

Adds:
- Random hide horizon during TRAINING
- ETA multitask learning (TFT only) OR two-stage ETA->Port mode (TFT only)
- Correct enc_len handling (true unpadded length)
- Split strategies: random or time (no MMSI-group split)
- obs_frac static feature (fraction of observed points after hide/clamp). Removable via --remove-statics obs_frac
"""

import os
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Assumes these exist in your project
from model import TFTClassifier
from transformer import ModelB_TransformerWithStatic
from LSTMandStatics import ModelA_BiLSTMWithStatic


# ---------- constants ----------
MIN_LAT, MAX_LAT = 69.2, 73.0
MIN_LON, MAX_LON = 13.0, 31.5
MIN_SOG, MAX_SOG = 0.0, 30.0
MIN_COG, MAX_COG = 0.0, 360.0

AIS_STEP_MINUTES = 5  # 5-minute steps in preprocessing

MASTER_STATIC_CONT_FEATURES = [
    "length_m",
    "draught_m",
    "engine_kw",
    "gross_tonnage",
    "total_catch_kg",
    "catch_num_species",
    "catch_entropy",
    "obs_frac",  # <-- fraction observed after hide; removable via --remove-statics obs_frac
]

MASTER_STATIC_CAT_FEATURES = [
    "gear",
    "mmsi",
    "season",
    "primary_species",
]


# ---------- data loading ----------

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
    Adds season and catch aggregates to voyages_df:
      - season
      - catch_num_species
      - catch_entropy
      - primary_species
    """
    voyages_df = voyages_df.copy()

    def _season_from_ts(ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return "Unknown"
        m = ts.month
        if m in (1, 2, 3, 4):
            return "Winter"
        elif m in (5, 6):
            return "Spring"
        elif m in (7, 8):
            return "Summer"
        else:
            return "Autumn"

    if "start_ts" in voyages_df.columns:
        ts_col = voyages_df["start_ts"].where(
            voyages_df["start_ts"].notna(),
            voyages_df.get("label_ts")
        )
    else:
        ts_col = voyages_df.get("label_ts")

    voyages_df["season"] = ts_col.apply(_season_from_ts)

    all_catches = []
    for y in years:
        path = os.path.join(data_dir, f"voyage_catches_{y}.csv")
        if os.path.exists(path):
            print(f"[INFO] Loading catches from {path}")
            c = pd.read_csv(path)
            all_catches.append(c)
        else:
            print(f"[INFO] No voyage_catches file for year {y} (path={path})")

    if not all_catches:
        print("[INFO] No voyage_catches files found – catch-based features defaulted.")
        voyages_df["catch_num_species"] = np.nan
        voyages_df["catch_entropy"] = np.nan
        if "primary_species" not in voyages_df.columns:
            voyages_df["primary_species"] = "Unknown"
        return voyages_df, None

    catches_df = pd.concat(all_catches, ignore_index=True)

    if "voyage_id" not in catches_df.columns or "total_rundvekt_kg" not in catches_df.columns:
        print("[WARN] catches missing required columns; skipping catch aggregates.")
        voyages_df["catch_num_species"] = np.nan
        voyages_df["catch_entropy"] = np.nan
        if "primary_species" not in voyages_df.columns:
            voyages_df["primary_species"] = "Unknown"
        return voyages_df, catches_df

    for col in ["art_fao_code", "art_fao", "art_fdir_code", "art_fdir"]:
        if col not in catches_df.columns:
            catches_df[col] = ""

    catches_df["total_rundvekt_kg"] = pd.to_numeric(
        catches_df["total_rundvekt_kg"], errors="coerce"
    ).fillna(0.0)

    num_species = (
        catches_df.groupby("voyage_id")["art_fao_code"]
        .nunique()
        .reset_index(name="catch_num_species")
    )

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

    def _primary_species_per_group(group: pd.DataFrame) -> str:
        agg = (
            group.groupby(
                ["art_fao_code", "art_fao", "art_fdir_code", "art_fdir"],
                dropna=False
            )["total_rundvekt_kg"]
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

    voyages_df = voyages_df.merge(num_species, on="voyage_id", how="left")
    voyages_df = voyages_df.merge(entropy, on="voyage_id", how="left")
    voyages_df = voyages_df.merge(primary_species, on="voyage_id", how="left")

    if "primary_species" not in voyages_df.columns:
        voyages_df["primary_species"] = voyages_df["primary_species_from_catches"]
    else:
        mask = voyages_df["primary_species"].isna() | (voyages_df["primary_species"].astype(str).str.len() == 0)
        voyages_df.loc[mask, "primary_species"] = voyages_df.loc[mask, "primary_species_from_catches"]

    voyages_df.drop(columns=["primary_species_from_catches"], inplace=True, errors="ignore")
    voyages_df["primary_species"] = voyages_df["primary_species"].fillna("Unknown").astype(str)

    return voyages_df, catches_df


# ---------- split helpers ----------

def split_voyage_ids(
    voyages_df: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    strategy: str = "random",
    time_col: str = "start_ts",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns train_ids, val_ids, test_ids

    strategy:
      - random: random over voyage_id
      - time: sort by time_col (fallback label_ts), oldest -> newest split
    """
    assert 0 < train_frac < 1
    assert 0 < val_frac < 1
    assert train_frac + val_frac < 1

    df = voyages_df.copy()
    df = df[df["voyage_id"].notna()].copy()

    rng = np.random.default_rng(seed)

    if strategy == "random":
        vids = df["voyage_id"].dropna().unique().tolist()
        rng.shuffle(vids)
        n_total = len(vids)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        return vids[:n_train], vids[n_train:n_train + n_val], vids[n_train + n_val:]

    if strategy == "time":
        if time_col not in df.columns:
            time_col = "label_ts"
        if time_col not in df.columns:
            raise ValueError("time split requested but no suitable time column found (start_ts/label_ts).")

        t = pd.to_datetime(df[time_col], errors="coerce")
        if "label_ts" in df.columns:
            t = t.where(t.notna(), pd.to_datetime(df["label_ts"], errors="coerce"))
        df["_split_time"] = t

        df = df.sort_values("_split_time", na_position="last")
        vids = df["voyage_id"].tolist()

        n_total = len(vids)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        return vids[:n_train], vids[n_train:n_train + n_val], vids[n_train + n_val:]

    raise ValueError(f"Unknown split strategy: {strategy}")


# ---------- dataset ----------

class VoyagePortDataset(Dataset):
    """
    One sample = one voyage.

    Encoder inputs:
      AIS sequence lat/lon/sog/cog normalized, length=encoder_length
      from observed part of track (end may be hidden).

    Static inputs:
      continuous + categorical subsets

    Targets:
      - destination port class
      - ETA target (remaining minutes or log1p(remaining_minutes) at cutoff)
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
        # random hide for training robustness
        train_random_hide: bool = False,
        train_hide_min_hours: float = 0.0,
        train_hide_max_hours: float = 0.0,
        seed: int = 42,
        # ETA target config
        eta_target_log1p: bool = True,
        eta_clip_minutes: float = 24.0 * 60.0,
    ):
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
        self.hide_last_n_points = int(max(hide_last_n_points, 0))
        self.window_mode = window_mode
        self.name = name

        self.train_random_hide = train_random_hide
        self.train_hide_min_hours = float(train_hide_min_hours)
        self.train_hide_max_hours = float(train_hide_max_hours)
        self.rng = np.random.default_rng(seed)

        self.eta_target_log1p = eta_target_log1p
        self.eta_clip_minutes = float(max(eta_clip_minutes, 0.0))

        # index AIS by voyage_id
        self.ais_grouped = ais_df.sort_values("ts").groupby("voyage_id")

        # merge voyages + vessels to collect static fields
        merged = voyages_df.merge(
            vessels_df,
            on="mmsi",
            how="left",
            suffixes=("", "_vessel"),
        )

        if voyage_ids is not None:
            voyage_ids_set = set(voyage_ids)
            merged = merged[merged["voyage_id"].isin(voyage_ids_set)]

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

            df_traj_full = self.ais_grouped.get_group(vid)
            if len(df_traj_full) == 0:
                continue

            # quick validity check using fixed hide
            n = len(df_traj_full)
            hide_n_check = min(self.hide_last_n_points, max(n - 1, 0))
            if hide_n_check > 0:
                df_obs = df_traj_full.iloc[:-hide_n_check]
            else:
                df_obs = df_traj_full

            if len(df_obs) == 0:
                continue

            valid_rows.append(row)

        self.voyages = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(
            f"[Dataset] Using {len(self.voyages)} voyages from {len(merged)} raw voyages "
            f"({self.name}, window_mode={self.window_mode}, train_random_hide={self.train_random_hide})"
        )

    def __len__(self):
        return len(self.voyages)

    def _normalize_track(self, df_traj: pd.DataFrame):
        """
        window_mode == "tail": use LAST encoder_length points; pad at BEGINNING
        window_mode == "head": use FIRST encoder_length points; pad at END

        IMPORTANT:
        - We return enc_len = true unpadded length (<= encoder_length).
        - Padding values are not special-cased in the model; enc_len is how the LSTM pack avoids them.
        """
        if self.window_mode == "tail":
            df_traj = df_traj.tail(self.encoder_length)
        else:
            df_traj = df_traj.head(self.encoder_length)

        true_len = len(df_traj)  # <-- unpadded length
        pad_len = self.encoder_length - true_len

        lat = df_traj["lat"].to_numpy()
        lon = df_traj["lon"].to_numpy()
        sog = df_traj["sog"].to_numpy()
        cog = (df_traj["cog"].to_numpy() % 360.0)

        if self.window_mode == "tail":
            if pad_len > 0:
                lat = np.concatenate([np.full(pad_len, MIN_LAT, dtype=float), lat])
                lon = np.concatenate([np.full(pad_len, MIN_LON, dtype=float), lon])
                sog = np.concatenate([np.zeros(pad_len, dtype=float), sog])
                cog = np.concatenate([np.zeros(pad_len, dtype=float), cog])
        else:  # head
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

        lat_t = torch.tensor(lat_n, dtype=torch.float32).unsqueeze(-1)
        lon_t = torch.tensor(lon_n, dtype=torch.float32).unsqueeze(-1)
        sog_t = torch.tensor(sog_n, dtype=torch.float32).unsqueeze(-1)
        cog_t = torch.tensor(cog_n, dtype=torch.float32).unsqueeze(-1)

        enc_len = torch.tensor(true_len, dtype=torch.long)  # <-- correct
        return {"lat": lat_t, "lon": lon_t, "sog": sog_t, "cog": cog_t}, enc_len

    def __getitem__(self, idx):
        row = self.voyages.iloc[idx]
        vid = row["voyage_id"]
        port_id = int(row["label_port_id"])
        port_class = self.port_id_to_idx[port_id]

        df_traj_full = self.ais_grouped.get_group(vid).sort_values("ts")

        # choose hide
        hide_n = self.hide_last_n_points
        if self.train_random_hide:
            h = float(self.rng.uniform(self.train_hide_min_hours, self.train_hide_max_hours))
            points_per_hour = int(60 / AIS_STEP_MINUTES)  # 12
            hide_n = int(round(h * points_per_hour))

        # clamp hide so at least one observed point remains
        n_full = len(df_traj_full)
        hide_n = min(max(hide_n, 0), max(n_full - 1, 0))

        if hide_n > 0:
            df_traj_obs = df_traj_full.iloc[:-hide_n]
        else:
            df_traj_obs = df_traj_full

        enc_vars, enc_len = self._normalize_track(df_traj_obs)

        # decoder dummy step (for TFT)
        dec_vars = {"dec_time": torch.zeros(1, 1, dtype=torch.float32)}

        # obs_frac (fraction observed after hide/clamp)
        obs_frac = float(len(df_traj_obs) / max(len(df_traj_full), 1))

        # static continuous (include obs_frac if in static_cont_features)
        static_cont_vals = []
        for c in self.static_cont_features:
            if c == "obs_frac":
                static_cont_vals.append(obs_frac)
                continue
            val = row.get(c, np.nan)
            static_cont_vals.append(0.0 if pd.isna(val) else float(val))
        static_cont = torch.tensor(static_cont_vals, dtype=torch.float32)

        # static categorical
        static_cat_vals: List[int] = []
        for feat in self.static_cat_features:
            if feat == "gear":
                gear = str(row.get("gear_primary", "Unknown"))
                static_cat_vals.append(self.gear_to_idx.get(gear, self.gear_to_idx["Unknown"]))
            elif feat == "mmsi":
                mmsi_str = str(row.get("mmsi", "Unknown"))
                static_cat_vals.append(self.mmsi_to_idx.get(mmsi_str, self.mmsi_to_idx["Unknown"]))
            elif feat == "season":
                season_str = str(row.get("season", "Unknown"))
                static_cat_vals.append(self.season_to_idx.get(season_str, self.season_to_idx["Unknown"]))
            elif feat == "primary_species":
                species_str = str(row.get("primary_species", "Unknown"))
                static_cat_vals.append(self.species_to_idx.get(species_str, self.species_to_idx["Unknown"]))
            else:
                raise ValueError(f"Unknown static categorical feature: {feat}")

        static_cat = torch.tensor(static_cat_vals, dtype=torch.long)
        target = torch.tensor(port_class, dtype=torch.long)

        # ETA target = remaining minutes from cutoff to last AIS timestamp we have
        if len(df_traj_obs) > 0 and len(df_traj_full) > 0:
            t_obs_last = df_traj_obs["ts"].iloc[-1]
            t_full_last = df_traj_full["ts"].iloc[-1]
            remaining_minutes = max((t_full_last - t_obs_last).total_seconds() / 60.0, 0.0)
        else:
            remaining_minutes = 0.0

        if self.eta_clip_minutes > 0:
            remaining_minutes = min(remaining_minutes, self.eta_clip_minutes)

        if self.eta_target_log1p:
            eta_target_val = np.log1p(remaining_minutes)
        else:
            eta_target_val = remaining_minutes

        eta_target = torch.tensor(float(eta_target_val), dtype=torch.float32)

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

    return (
        enc_vars_batch,
        dec_vars_batch,
        torch.stack(enc_lens, dim=0),
        torch.stack(static_conts, dim=0),
        torch.stack(static_cats, dim=0),
        torch.stack(targets, dim=0),
        torch.stack(eta_targets, dim=0),
    )


# ---------- ETA losses ----------

def pinball_loss(y_pred_q: torch.Tensor, y_true: torch.Tensor, quantiles: Tuple[float, ...]) -> torch.Tensor:
    """
    y_pred_q: (B, Q)
    y_true:   (B,)
    """
    q = torch.tensor(quantiles, device=y_pred_q.device, dtype=y_pred_q.dtype).view(1, -1)
    e = y_true.unsqueeze(-1) - y_pred_q
    return torch.maximum(q * e, (q - 1.0) * e).mean()


def eta_loss_fn(
    eta_pred_q: torch.Tensor,
    eta_target: torch.Tensor,
    quantiles: Tuple[float, ...],
    loss_type: str = "quantile",
) -> torch.Tensor:
    """
    eta_pred_q: (B,Q) or (B,T,Q)
    eta_target: (B,)
    """
    if eta_pred_q is None:
        return torch.zeros((), device=eta_target.device)

    if eta_pred_q.dim() == 3:
        eta_pred_q = eta_pred_q[:, -1, :]

    if loss_type == "quantile":
        return pinball_loss(eta_pred_q, eta_target, quantiles)

    q_arr = np.asarray(quantiles, dtype=float)
    mid_idx = int(np.argmin(np.abs(q_arr - 0.5)))
    eta_med = eta_pred_q[:, mid_idx]

    if loss_type == "huber":
        return F.huber_loss(eta_med, eta_target, delta=1.0)
    elif loss_type == "mae":
        return F.l1_loss(eta_med, eta_target)
    else:
        raise ValueError(f"Unknown ETA loss type: {loss_type}")


def evaluate_eta_mae_minutes(
    model,
    loader,
    device,
    eta_target_log1p: bool,
) -> float:
    model.eval()
    if not hasattr(model, "eta_quantiles") or len(model.eta_quantiles) == 0:
        return float("nan")

    q_arr = np.asarray(model.eta_quantiles, dtype=float)
    mid_idx = int(np.argmin(np.abs(q_arr - 0.5)))

    total_abs = 0.0
    total_cnt = 0

    with torch.no_grad():
        for batch in loader:
            enc_vars, dec_vars, enc_lens, static_cont, static_cat, _, eta_targets = batch

            enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
            dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
            enc_lens = enc_lens.to(device)
            static_cont = static_cont.to(device)
            static_cat = static_cat.to(device)
            eta_targets = eta_targets.to(device)

            (logits_unused, eta_q), _ = model(
                enc_vars=enc_vars,
                dec_vars=dec_vars,
                enc_lengths=enc_lens,
                static_cont=static_cont,
                static_cat=static_cat,
            )

            if eta_q is None:
                continue
            if eta_q.dim() == 3:
                eta_q = eta_q[:, -1, :]

            eta_pred = eta_q[:, mid_idx]

            if eta_target_log1p:
                pred_min = torch.clamp(torch.expm1(eta_pred), min=0.0)
                true_min = torch.clamp(torch.expm1(eta_targets), min=0.0)
            else:
                pred_min = torch.clamp(eta_pred, min=0.0)
                true_min = torch.clamp(eta_targets, min=0.0)

            total_abs += torch.abs(pred_min - true_min).sum().item()
            total_cnt += true_min.numel()

    return total_abs / max(total_cnt, 1)


# ---------- two-stage ETA->Port helpers ----------

def _eta_mid_idx(quantiles: Tuple[float, ...]) -> int:
    q_arr = np.asarray(quantiles, dtype=float)
    return int(np.argmin(np.abs(q_arr - 0.5)))


@torch.no_grad()
def predict_eta_feature_logspace(
    eta_model,
    enc_vars,
    dec_vars,
    enc_lens,
    static_cont,
    static_cat,
) -> torch.Tensor:
    """
    Returns (B,1) ETA feature in *model output space* (e.g. log1p minutes if trained with log1p),
    using median quantile.
    """
    eta_model.eval()
    (_, eta_q), _ = eta_model(
        enc_vars=enc_vars,
        dec_vars=dec_vars,
        enc_lengths=enc_lens,
        static_cont=static_cont,
        static_cat=static_cat,
    )

    if eta_q is None:
        B = enc_lens.size(0)
        return torch.zeros(B, 1, device=enc_lens.device, dtype=torch.float32)

    if eta_q.dim() == 3:
        eta_q = eta_q[:, -1, :]  # (B,Q)

    mid = _eta_mid_idx(eta_model.eta_quantiles)
    return eta_q[:, mid].unsqueeze(1)  # (B,1)


def train_eta_only_model(
    eta_model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    eta_loss: str,
    eta_target_log1p: bool,
):
    """
    Train eta_model using ONLY ETA loss (no classification loss).
    Select best by val ETA MAE (minutes), then freeze model.
    """
    assert hasattr(eta_model, "eta_quantiles") and len(eta_model.eta_quantiles) > 0, \
        "ETA-only model must have eta_quantiles > 0"

    optimizer = torch.optim.Adam(eta_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None

    print("[ETA PRETRAIN] Training ETA-only model...")
    for epoch in range(1, epochs + 1):
        eta_model.train()
        total_eta_loss = 0.0
        total_cnt = 0

        for batch in train_loader:
            enc_vars, dec_vars, enc_lens, static_cont, static_cat, _, eta_targets = batch

            enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
            dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
            enc_lens = enc_lens.to(device)
            static_cont = static_cont.to(device)
            static_cat = static_cat.to(device)
            eta_targets = eta_targets.to(device)

            (_, eta_q), _ = eta_model(
                enc_vars=enc_vars,
                dec_vars=dec_vars,
                enc_lengths=enc_lens,
                static_cont=static_cont,
                static_cat=static_cat,
            )

            loss_eta = eta_loss_fn(
                eta_pred_q=eta_q,
                eta_target=eta_targets,
                quantiles=eta_model.eta_quantiles,
                loss_type=eta_loss,
            )

            optimizer.zero_grad()
            loss_eta.backward()
            optimizer.step()

            bsz = eta_targets.size(0)
            total_eta_loss += loss_eta.item() * bsz
            total_cnt += bsz

        train_eta = total_eta_loss / max(total_cnt, 1)
        val_mae = evaluate_eta_mae_minutes(
            model=eta_model,
            loader=val_loader,
            device=device,
            eta_target_log1p=eta_target_log1p,
        )

        print(f"[ETA PRETRAIN Epoch {epoch:02d}] train_eta_loss={train_eta:.4f}  val_eta_mae_min={val_mae:.2f}")

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in eta_model.state_dict().items()}

    if best_state is not None:
        eta_model.load_state_dict(best_state)
        print(f"[ETA PRETRAIN] Loaded best ETA model (val_eta_mae_min={best_val:.2f}).")

    # freeze
    for p in eta_model.parameters():
        p.requires_grad = False
    eta_model.eval()
    return eta_model


# ---------- evaluation ----------

def evaluate_model(model, loader, device, model_type: str, *, two_stage: bool = False, eta_model=None):
    model.eval()
    if eta_model is not None:
        eta_model.eval()

    total_loss = 0.0
    total_correct = 0
    total_cnt = 0

    with torch.no_grad():
        for batch in loader:
            enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, _ = batch

            enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
            dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
            enc_lens = enc_lens.to(device)
            static_cont = static_cont.to(device)
            static_cat = static_cat.to(device)
            targets = targets.to(device)


            if model_type == "tft":
                (logits, _), _ = model(
                    enc_vars=enc_vars,
                    dec_vars=dec_vars,
                    enc_lengths=enc_lens,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )
            else:
                x_seq = torch.cat([enc_vars["lat"], enc_vars["lon"], enc_vars["sog"], enc_vars["cog"]], dim=-1)
                logits = model(x_seq=x_seq, static_cont=static_cont, static_cat=static_cat)

            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_cnt += targets.size(0)

    return total_loss / max(total_cnt, 1), total_correct / max(total_cnt, 1)


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
    eta_target_log1p: bool,
    enc_feature_names=("lat", "lon", "sog", "cog"),
    top_k=5,
    model_type: str = "tft",
    *,
    two_stage: bool = False,
    eta_model=None,
    show_eta_if_available: bool = True,
):
    model.eval()
    if eta_model is not None:
        eta_model.eval()

    batch = next(iter(test_loader))
    enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, eta_targets = batch

    enc_vars_dev = {k: v.to(device) for k, v in enc_vars.items()}
    dec_vars_dev = {k: v.to(device) for k, v in dec_vars.items()}
    enc_lens_dev = enc_lens.to(device)
    static_cont_dev = static_cont.to(device)
    static_cat_dev = static_cat.to(device)
    targets_dev = targets.to(device)
    eta_targets_dev = eta_targets.to(device)

    eta_pred_min = None
    if eta_model is not None:
        with torch.no_grad():
            eta_feat = predict_eta_feature_logspace(
                eta_model=eta_model,
                enc_vars=enc_vars_dev,
                dec_vars=dec_vars_dev,
                enc_lens=enc_lens_dev,
                static_cont=static_cont_dev,
                static_cat=static_cat_dev,
            )  # (B,1) in log-space if log1p

        b = 0  # sample index we inspect
        eta_log = eta_feat[b, 0].item()

        if eta_target_log1p:
            eta_pred_min = max(np.expm1(eta_log), 0.0)
        else:
            eta_pred_min = max(eta_log, 0.0)

    with torch.no_grad():
        eta_q = None
        if model_type == "tft":
            (logits, eta_q), _ = model(
                enc_vars=enc_vars_dev,
                dec_vars=dec_vars_dev,
                enc_lengths=enc_lens_dev,
                static_cont=static_cont_dev,
                static_cat=static_cat_dev,
            )
        else:
            x_seq = torch.cat([enc_vars_dev["lat"], enc_vars_dev["lon"], enc_vars_dev["sog"], enc_vars_dev["cog"]], dim=-1)
            logits = model(x_seq=x_seq, static_cont=static_cont_dev, static_cat=static_cat_dev)

        probs = torch.softmax(logits, dim=-1)

    b = 0
    sample_probs = probs[b].cpu().numpy()
    true_idx = targets_dev[b].item()
    true_port_id = idx_to_port_id[true_idx]

    print("\n=== Inspecting one test sample ===")
    print(f"Batch size: {probs.size(0)}")
    print(f"True class index: {true_idx}  -> port_id={true_port_id}")

    if eta_pred_min is not None:
        print(f"Predicted ETA (median): {eta_pred_min:.1f} min")

    print("\nEncoder inputs (last 5 timesteps):")
    for feat in enc_feature_names:
        series = enc_vars[feat][b, :, 0].cpu().numpy()
        print(f"  {feat}: {series[-5:]}")

    sc = static_cont[b].cpu().numpy().tolist()
    print("\nStatic continuous (raw vector):")
    for i, v in enumerate(sc):
        print(f"  cont[{i}]: {v:.3f}")


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

    # print ETA prediction if the *port model* has an ETA head (single-model multitask)
    if show_eta_if_available and (eta_q is not None) and hasattr(model, "eta_quantiles") and len(model.eta_quantiles) > 0:
        if eta_q.dim() == 3:
            eta_q = eta_q[:, -1, :]
        qs = np.asarray(model.eta_quantiles, dtype=float)
        mid_idx = int(np.argmin(np.abs(qs - 0.5)))

        eta_pred_raw = eta_q[b, mid_idx].item()
        eta_true_raw = eta_targets_dev[b].item()

        if eta_target_log1p:
            eta_pred_min = max(np.expm1(eta_pred_raw), 0.0)
            eta_true_min = max(np.expm1(eta_true_raw), 0.0)
        else:
            eta_pred_min = max(eta_pred_raw, 0.0)
            eta_true_min = max(eta_true_raw, 0.0)

        print(f"\nETA (median q) prediction: {eta_pred_min:.1f} min, true: {eta_true_min:.1f} min")

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
        help="Fixed hide horizon for VAL/TEST (and TRAIN too if --train-random-hide is off).",
    )

    parser.add_argument(
        "--encoder-window-mode",
        type=str,
        choices=["tail", "head"],
        default="tail",
        help="'tail' uses last encoder_length points (pad at beginning); 'head' uses first points (pad at end).",
    )

    parser.add_argument(
        "--remove-statics",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Static features to drop. Allowed: "
            "length_m, draught_m, engine_kw, gross_tonnage, total_catch_kg, "
            "catch_num_species, catch_entropy, obs_frac, "
            "gear, mmsi, season, primary_species"
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tft", "bilstm", "transformer"],
        default="tft",
        help="Which architecture to train.",
    )

    # random hide during training
    parser.add_argument("--train-random-hide", action="store_true",
                        help="If set, sample hide horizon randomly per TRAIN sample.")
    parser.add_argument("--train-hide-min-hours", type=float, default=0.0)
    parser.add_argument("--train-hide-max-hours", type=float, default=6.0)

    # split strategy (no mmsi_group)
    parser.add_argument("--split-strategy", type=str, choices=["time", "random"], default="random")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--split-time-col", type=str, default="start_ts")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=6.34e-6)
    parser.add_argument("--seed", type=int, default=42)

    # ETA multitask (single-model, TFT only)
    parser.add_argument("--eta-multitask", action="store_true",
                        help="Enable ETA auxiliary task (TFT only).")
    parser.add_argument("--eta-loss-weight", type=float, default=0.2,
                        help="Total loss = CE + eta_loss_weight * ETA_loss")
    parser.add_argument("--eta-loss", type=str, choices=["quantile", "huber", "mae"], default="quantile")
    parser.add_argument("--eta-quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--eta-target-log1p", dest="eta_target_log1p", action="store_true")
    parser.add_argument("--no-eta-target-log1p", dest="eta_target_log1p", action="store_false")
    parser.set_defaults(eta_target_log1p=True)
    parser.add_argument("--eta-clip-hours", type=float, default=24.0)

    # Two-stage ETA -> Port (TFT only)
    parser.add_argument(
        "--two-stage-eta-port",
        action="store_true",
        help="Train a separate ETA model first, then train the port model"
    )
    parser.add_argument(
        "--eta-pretrain-epochs",
        type=int,
        default=20,
        help="Epochs for ETA-only pretraining when --two-stage-eta-port is enabled."
    )
    parser.add_argument(
        "--eta-pretrain-lr",
        type=float,
        default=None,
        help="LR for ETA pretraining. If None, uses --lr."
    )

    parser.add_argument(
        "--sweep-hide-hours",
        action="store_true",
        help="After training, evaluate TEST accuracy across multiple hide horizons and save plot.",
    )
    parser.add_argument("--sweep-hide-start", type=float, default=2.0)
    parser.add_argument("--sweep-hide-end", type=float, default=6.0)
    parser.add_argument("--sweep-hide-step", type=float, default=0.5)

    args = parser.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1.")

    if args.train_random_hide and args.train_hide_max_hours < args.train_hide_min_hours:
        raise ValueError("--train-hide-max-hours must be >= --train-hide-min-hours.")

    if args.model_type != "tft" and (args.eta_multitask or args.two_stage_eta_port):
        print("[WARN] ETA features only implemented for TFT. Disabling ETA features.")
        args.eta_multitask = False
        args.two_stage_eta_port = False

    if args.two_stage_eta_port and args.eta_multitask:
        print("[WARN] Both --two-stage-eta-port and --eta-multitask set; using two-stage and disabling multitask.")
        args.eta_multitask = False

    if args.eta_multitask and len(args.eta_quantiles) == 0:
        raise ValueError("ETA multitask enabled but --eta-quantiles is empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model type: {args.model_type}")

    print(f"[INFO] Loading data from {args.data_dir} for years {args.years}")
    ais_df, voyages_df, vessels_df = load_all_years(args.data_dir, args.years)

    # add season + catch-based features
    voyages_df, _ = add_season_and_catch_features(args.data_dir, args.years, voyages_df)

    # ---- mappings ----
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

    # ---- static features ----
    remove_set = set(args.remove_statics or [])
    static_cont_features = [f for f in MASTER_STATIC_CONT_FEATURES if f not in remove_set]
    static_cat_features = [f for f in MASTER_STATIC_CAT_FEATURES if f not in remove_set]

    print(f"[INFO] Using static continuous features: {static_cont_features}")
    print(f"[INFO] Using static categorical features: {static_cat_features}")

    # ---- split voyage IDs ----
    train_ids, val_ids, test_ids = split_voyage_ids(
        voyages_df=voyages_df,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        strategy=args.split_strategy,
        time_col=args.split_time_col,
    )
    print(f"[INFO] Split strategy: {args.split_strategy}")
    print(f"[INFO] Voyage split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ---- fixed hide points ----
    if args.hide_last_hours > 0:
        points_per_hour = int(60 / AIS_STEP_MINUTES)  # 12
        hide_last_n_points = int(round(args.hide_last_hours * points_per_hour))
    else:
        hide_last_n_points = 0

    print(f"[INFO] Fixed hide_last_hours={args.hide_last_hours} -> hide_last_n_points={hide_last_n_points}")
    if args.train_random_hide:
        print(f"[INFO] TRAIN random hide in [{args.train_hide_min_hours}, {args.train_hide_max_hours}] hours")

    # ---- datasets ----
    dataset_kwargs_common = dict(
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
        window_mode=args.encoder_window_mode,
        eta_target_log1p=args.eta_target_log1p,
        eta_clip_minutes=args.eta_clip_hours * 60.0,
    )

    train_dataset = VoyagePortDataset(
        **dataset_kwargs_common,
        hide_last_n_points=hide_last_n_points,  # fallback if not random hide
        voyage_ids=train_ids,
        name="train",
        train_random_hide=args.train_random_hide,
        train_hide_min_hours=args.train_hide_min_hours,
        train_hide_max_hours=args.train_hide_max_hours,
        seed=args.seed,
    )

    val_dataset = VoyagePortDataset(
        **dataset_kwargs_common,
        hide_last_n_points=hide_last_n_points,
        voyage_ids=val_ids,
        name="val",
        train_random_hide=False,
        seed=args.seed + 1,
    )

    test_dataset = VoyagePortDataset(
        **dataset_kwargs_common,
        hide_last_n_points=hide_last_n_points,
        voyage_ids=test_ids,
        name="test",
        train_random_hide=False,
        seed=args.seed + 2,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # ---- model ----
    enc_var_dims = {"lat": 1, "lon": 1, "sog": 1, "cog": 1}
    dec_var_dims = {"dec_time": 1}

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

    eta_model = None

    if args.model_type == "tft":
        if args.two_stage_eta_port:
            # ETA model: needs ETA head
            eta_qs = tuple(args.eta_quantiles) if len(args.eta_quantiles) > 0 else (0.1, 0.5, 0.9)
            eta_model = TFTClassifier(
                enc_var_dims=enc_var_dims,
                dec_var_dims=dec_var_dims,
                static_cont_dim=len(static_cont_features),
                static_cat_cardinalities=static_cat_cardinalities,
                static_cat_emb_dim=16,
                d_model=128,
                lstm_layers=2,
                dropout=0.15,
                num_heads=4,
                num_classes=num_ports,  # unused for ETA pretrain
                pred_len=1,
                eta_quantiles=eta_qs,
            ).to(device)

            static_cont_dim_port = len(static_cont_features)
            model = TFTClassifier(
                enc_var_dims=enc_var_dims,
                dec_var_dims=dec_var_dims,
                static_cont_dim=static_cont_dim_port,
                static_cat_cardinalities=static_cat_cardinalities,
                static_cat_emb_dim=16,
                d_model=128,
                lstm_layers=2,
                dropout=0.15,
                num_heads=4,
                num_classes=num_ports,
                pred_len=1,
                eta_quantiles=(),
            ).to(device)

        else:
            # single model (optional multitask)
            eta_qs = tuple(args.eta_quantiles) if args.eta_multitask else ()
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
                eta_quantiles=eta_qs,
            ).to(device)

    elif args.model_type == "bilstm":
        model = ModelA_BiLSTMWithStatic(
            ts_input_dim=4,
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

    # ---- optional ETA pretrain (two-stage) ----
    if args.two_stage_eta_port:
        pre_lr = args.lr if args.eta_pretrain_lr is None else args.eta_pretrain_lr
        eta_model = train_eta_only_model(
            eta_model=eta_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.eta_pretrain_epochs,
            lr=pre_lr,
            weight_decay=args.weight_decay,
            eta_loss=args.eta_loss,
            eta_target_log1p=args.eta_target_log1p,
        )
        print("[INFO] Two-stage mode enabled: ETA model is trained separately; port model is trained independently.")

    # ---- training ----
    best_val_acc = 0.0
    best_state = None

    print("[TRAIN] Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_eta_loss = 0.0
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
            eta_targets = eta_targets.to(device)


            if args.model_type == "tft":
                (logits, eta_q), _ = model(
                    enc_vars=enc_vars,
                    dec_vars=dec_vars,
                    enc_lengths=enc_lens,
                    static_cont=static_cont,
                    static_cat=static_cat,
                )
            else:
                x_seq = torch.cat([enc_vars["lat"], enc_vars["lon"], enc_vars["sog"], enc_vars["cog"]], dim=-1)
                logits = model(x_seq=x_seq, static_cont=static_cont, static_cat=static_cat)
                eta_q = None

            loss_cls = F.cross_entropy(logits, targets)
            loss = loss_cls

            loss_eta = torch.zeros((), device=device)
            if (not args.two_stage_eta_port) and args.eta_multitask and (args.model_type == "tft") and (eta_q is not None):
                loss_eta = eta_loss_fn(
                    eta_pred_q=eta_q,
                    eta_target=eta_targets,
                    quantiles=model.eta_quantiles,
                    loss_type=args.eta_loss,
                )
                loss = loss_cls + args.eta_loss_weight * loss_eta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = targets.size(0)
            total_loss += loss.item() * bsz
            total_cls_loss += loss_cls.item() * bsz
            if (not args.two_stage_eta_port) and args.eta_multitask and (args.model_type == "tft"):
                total_eta_loss += loss_eta.item() * bsz

            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_cnt += bsz

        train_loss = total_loss / max(total_cnt, 1)
        train_cls_loss = total_cls_loss / max(total_cnt, 1)
        train_eta_loss = total_eta_loss / max(total_cnt, 1) if ((not args.two_stage_eta_port) and args.eta_multitask and args.model_type == "tft") else float("nan")
        train_acc = total_correct / max(total_cnt, 1)

        val_loss, val_acc = evaluate_model(
            model, val_loader, device, args.model_type,
            two_stage=args.two_stage_eta_port, eta_model=eta_model
        )

        log_msg = (
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_cls={train_cls_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if (not args.two_stage_eta_port) and args.eta_multitask and args.model_type == "tft":
            val_eta_mae_min = evaluate_eta_mae_minutes(
                model=model,
                loader=val_loader,
                device=device,
                eta_target_log1p=args.eta_target_log1p,
            )
            log_msg += f"  train_eta={train_eta_loss:.4f} val_eta_mae_min={val_eta_mae_min:.2f}"

        print(log_msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("[TRAIN] Training finished.")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Loaded best model (val_acc={best_val_acc:.3f}) for final test evaluation.")

    test_loss, test_acc = evaluate_model(
        model, test_loader, device, args.model_type,
        two_stage=args.two_stage_eta_port, eta_model=eta_model
    )
    print(f"[TEST] loss={test_loss:.4f}  acc={test_acc:.3f}")

    if (not args.two_stage_eta_port) and args.eta_multitask and args.model_type == "tft":
        test_eta_mae_min = evaluate_eta_mae_minutes(
            model=model,
            loader=test_loader,
            device=device,
            eta_target_log1p=args.eta_target_log1p,
        )
        print(f"[TEST ETA] MAE_minutes={test_eta_mae_min:.2f}")

    # ---------- feature importance (TFT only) ----------
    if args.model_type == "tft":
        model.eval()

        # encoder variable importance
        enc_alpha_sum = None
        n_enc_batches = 0
        enc_feature_names = list(enc_var_dims.keys())

        with torch.no_grad():
            for batch in test_loader:
                enc_vars, dec_vars, enc_lens, static_cont, static_cat, _, _ = batch
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
                batch_mean = enc_alpha.mean(dim=(0, 1))
                enc_alpha_sum = batch_mean if enc_alpha_sum is None else (enc_alpha_sum + batch_mean)
                n_enc_batches += 1

        if enc_alpha_sum is not None and n_enc_batches > 0:
            enc_alpha_mean = enc_alpha_sum / n_enc_batches
            print("\n[Feature importance] Encoder variables (test-set average):")
            for name, score in zip(enc_feature_names, enc_alpha_mean.cpu().numpy()):
                print(f"   {name}: {score:.4f}")
        else:
            print("\n[Feature importance] Encoder variables: enc_alpha not found in extras.")

        # static variable importance
        static_alpha_sum = None
        n_static_batches = 0

        ordered_static_names = static_cont_features + static_cat_features

        with torch.no_grad():
            for batch in test_loader:
                enc_vars, dec_vars, enc_lens, static_cont, static_cat, _, _ = batch
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
                batch_mean = static_alpha.mean(dim=0)
                static_alpha_sum = batch_mean if static_alpha_sum is None else (static_alpha_sum + batch_mean)
                n_static_batches += 1

        if static_alpha_sum is not None and n_static_batches > 0 and len(ordered_static_names) > 0:
            static_alpha_mean = static_alpha_sum / n_static_batches
            print("\n[Feature importance] STATIC variables (test-set average):")
            for name, score in zip(ordered_static_names, static_alpha_mean.cpu().numpy()):
                print(f"   {name}: {score:.4f}")
        else:
            print("\n[Feature importance] STATIC variables: either none in use or static_alpha not found.")
    else:
        print("\n[Feature importance] Skipped: only implemented for TFT.")

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
        eta_target_log1p=args.eta_target_log1p,
        enc_feature_names=list(enc_var_dims.keys()),
        top_k=5,
        model_type=args.model_type,
        two_stage=args.two_stage_eta_port,
        eta_model=eta_model,
        show_eta_if_available=True,
    )

    # ---------- sweep inference accuracy vs hidden-last-hours ----------
    if args.sweep_hide_hours:
        import time

        print("\n[SWEEP] Evaluating accuracy vs hide_last_hours on TEST set...")

        sweep_hours = np.arange(
            args.sweep_hide_start,
            args.sweep_hide_end + 1e-9,
            args.sweep_hide_step,
            dtype=float,
        )

        sweep_acc = []
        sweep_loss = []

        points_per_hour = int(60 / AIS_STEP_MINUTES)

        for h in sweep_hours:
            hide_last_n_points_sweep = int(round(h * points_per_hour))

            test_dataset_sweep = VoyagePortDataset(
                **dataset_kwargs_common,
                hide_last_n_points=hide_last_n_points_sweep,
                voyage_ids=test_ids,
                name=f"test_sweep_hide_{h:.1f}h",
                train_random_hide=False,
                seed=args.seed + 123,
            )

            test_loader_sweep = DataLoader(
                test_dataset_sweep,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_batch,
            )

            t0 = time.perf_counter()
            loss_h, acc_h = evaluate_model(
                model, test_loader_sweep, device, args.model_type,
                two_stage=args.two_stage_eta_port, eta_model=eta_model
            )
            dt = time.perf_counter() - t0

            sweep_loss.append(loss_h)
            sweep_acc.append(acc_h)

            print(f"  hide_last_hours={h:.1f} -> acc={acc_h:.4f}, loss={loss_h:.4f} (eval_time={dt:.2f}s)")

        plt.figure()
        plt.plot(sweep_hours, sweep_acc, marker="o")
        plt.xlabel("Hidden last hours (hours)")
        plt.ylabel("Test accuracy")
        plt.title(f"Accuracy vs hidden-last-hours (model={args.model_type})")
        plt.grid(True)

        out_path = os.path.join(
            args.data_dir,
            f"sweep_acc_vs_hidehours_{args.model_type}_twoStage{int(args.two_stage_eta_port)}.png"
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[SWEEP] Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
