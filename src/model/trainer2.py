"""
Train your custom TFTClassifier to predict destination port from AIS voyages.

Assumes preprocessed CSVs per year in --data-dir:

    ais_points_YYYY.csv
    voyages_YYYY.csv
    vessels_YYYY.csv

Usage example:

    python src/model/train_tft_ports.py \
        --data-dir output --years 2016 2017 2018 \
        --encoder-length 64 \
        --hide-last-hours 3 \
        --encoder-window-mode tail \
        --batch-size 64 \
        --epochs 30 \
        --remove-statics mmsi
"""

import os
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import TFTClassifier  # your custom TFT implementation


# ---------- constants: same region as preprocessing ----------
MIN_LAT, MAX_LAT = 69.2, 73.0
MIN_LON, MAX_LON = 13.0, 31.5
MIN_SOG, MAX_SOG = 0.0, 30.0
MIN_COG, MAX_COG = 0.0, 360.0

AIS_STEP_MINUTES = 5  # 5-minute steps in your preprocessing

# master static feature names; we will filter these based on CLI
MASTER_STATIC_CONT_FEATURES = ["length_m", "draught_m", "engine_kw", "gross_tonnage"]
MASTER_STATIC_CAT_FEATURES  = ["gear", "mmsi"]  # gear + vessel identity


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


# ---------- dataset ----------

class VoyagePortDataset(Dataset):
    """
    One sample = one voyage.

    Encoder inputs:
        - AIS sequence: lat/lon/sog/cog normalized, length = encoder_length,
          taken from the **observed part** of the track (last N hours can be hidden).

    Static inputs:
        - continuous: subset of [length_m, draught_m, engine_kw, gross_tonnage]
        - categorical: subset of [gear_primary, mmsi]

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
            - pad at the END if the sequence is shorter.
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

        # Decoder input: single dummy time step
        T_dec = 1
        dec_vars = {
            "dec_time": torch.zeros(T_dec, 1, dtype=torch.float32)
        }

        # Static continuous: subset of [length_m, draught_m, engine_kw, gross_tonnage]
        static_cont_vals = []
        for c in self.static_cont_features:
            val = row.get(c, np.nan)
            static_cont_vals.append(0.0 if pd.isna(val) else float(val))
        static_cont = torch.tensor(static_cont_vals, dtype=torch.float32)  # (n_cont,)

        # Static categorical: subset of [gear_primary, mmsi]
        static_cat_vals = []

        if "gear" in self.static_cat_features:
            gear = str(row.get("gear_primary", "Unknown"))
            gear_idx = self.gear_to_idx.get(gear, self.gear_to_idx["Unknown"])
            static_cat_vals.append(gear_idx)

        if "mmsi" in self.static_cat_features:
            mmsi_str = str(row.get("mmsi", "Unknown"))
            mmsi_idx = self.mmsi_to_idx.get(mmsi_str, self.mmsi_to_idx["Unknown"])
            static_cat_vals.append(mmsi_idx)

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

def evaluate_model(model, loader, device):
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

            (logits, eta_q), extras = model(
                enc_vars=enc_vars,
                dec_vars=dec_vars,
                enc_lengths=enc_lens,
                static_cont=static_cont,
                static_cat=static_cat,
            )
            logits = logits.squeeze(1)
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
    enc_feature_names=("lat", "lon", "sog", "cog"),
    top_k=5,
):
    """
    Take one batch from test_loader, inspect the first sample:
      - print some of the encoder inputs
      - print static features (length, GT, gear, mmsi)
      - print top-k predicted ports with probabilities
    """
    model.eval()
    batch = next(iter(test_loader))

    enc_vars, dec_vars, enc_lens, static_cont, static_cat, targets, eta_targets = batch

    # Move to device
    enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
    dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
    enc_lens = enc_lens.to(device)
    static_cont = static_cont.to(device)
    static_cat = static_cat.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        (logits, eta_q), extras = model(
            enc_vars=enc_vars,
            dec_vars=dec_vars,
            enc_lengths=enc_lens,
            static_cont=static_cont,
            static_cat=static_cat,
        )

        # logits: (B, 1, num_ports)
        logits = logits.squeeze(1)  # (B, num_ports)
        probs = torch.softmax(logits, dim=-1)  # (B, num_ports)

    # Inspect only first sample in this batch
    b = 0

    sample_probs = probs[b].cpu().numpy()
    true_idx = targets[b].item()
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
    # STATIC_CONT_FEATURES = ["length_m", "draught_m", "engine_kw", "gross_tonnage"]
    length_m, draught_m, engine_kw, gross_tonnage = static_cont[b].cpu().numpy().tolist()
    print("\nStatic continuous:")
    print(f"  length_m:      {length_m:.2f}")
    print(f"  draught_m:     {draught_m:.2f}")
    print(f"  engine_kw:     {engine_kw:.2f}")
    print(f"  gross_tonnage: {gross_tonnage:.2f}")

    # ----- static categorical (gear, mmsi) -----
    gear_idx = static_cat[b, 0].item()
    mmsi_idx = static_cat[b, 1].item()
    gear_str = idx_to_gear.get(gear_idx, f"<unknown-{gear_idx}>")
    mmsi_str = idx_to_mmsi.get(mmsi_idx, f"<unknown-{mmsi_idx}>")

    print("\nStatic categorical:")
    print(f"  gear: {gear_str} (idx={gear_idx})")
    print(f"  mmsi: {mmsi_str} (idx={mmsi_idx})")

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
    parser = argparse.ArgumentParser(description="Train custom TFT to predict destination port with partial tracks")
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
            "Allowed: length_m, draught_m, engine_kw, gross_tonnage, gear, mmsi"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading data from {args.data_dir} for years {args.years}")
    ais_df, voyages_df, vessels_df = load_all_years(args.data_dir, args.years)

    # ---- port / gear / mmsi mappings ----
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

    idx_to_port_id = {idx: pid for pid, idx in port_id_to_idx.items()}
    idx_to_gear = {idx: g for g, idx in gear_to_idx.items()}
    idx_to_mmsi = {idx: m for m, idx in mmsi_to_idx.items()}

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
        else:
            raise ValueError(f"Unknown categorical static feature: {f}")

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

            (logits, eta_q), extras = model(
                enc_vars=enc_vars,
                dec_vars=dec_vars,
                enc_lengths=enc_lens,
                static_cont=static_cont,
                static_cat=static_cat,
            )
            logits = logits.squeeze(1)
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

        val_loss, val_acc = evaluate_model(model, val_loader, device)

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

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"[TEST] loss={test_loss:.4f}  acc={test_acc:.3f}")

    # ---------- feature importance on ENCODER variables ----------
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
        
    
    inspect_one_test_sample(
    model=model,
    test_loader=test_loader,
    device=device,
    idx_to_port_id=idx_to_port_id,
    idx_to_gear=idx_to_gear,
    idx_to_mmsi=idx_to_mmsi,
    enc_feature_names=list(enc_var_dims.keys()),
    top_k=5,
)


if __name__ == "__main__":
    main()
