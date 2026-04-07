#!/usr/bin/env python3
import os, uuid, argparse, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from bisect import bisect_right
from pyproj import Geod
from tqdm import tqdm
import re

# Optional database imports - only needed for DB mode
try:
    from psycopg import connect
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    print("[!] Warning: psycopg not available. Database mode will not work.")
    print("[!] Install with: pip install psycopg[binary]")

"""
AIS-ERS preprocessing + catches:
- Voyage windows / AIS track selection now matches the OLD script logic:
    * starttime = Stopptidspunkt
    * stoptime  = Ankomsttidspunkt + (Avgangstidspunkt - Ankomsttidspunkt)/2
    * Filter AIS by MMSI and within (starttime, stoptime)
    * Gap filter, 20-msg + 4h minimum, port within 5km via 'coords' column
    * Downsample + normalize
- On top of that we attach:
    * Static info from ERS
    * Gear usage timeline
    * Catches (DCA) per voyage/species
    * CSV outputs for vessels/voyages/ais_points/catches
"""

geod = Geod(ellps='WGS84')

# ---------------- normalization bounds (region-specific) ----------------
min_cog, max_cog = 0.0, 360.0
min_sog, max_sog = 0, 30.0
min_lat, max_lat = 69.2, 73.0
min_lon, max_lon = 13.0, 31.5

# Require end-of-window to be within N km of a port (old script used 5 km)
PORT_MAX_KM = 5.0

# ---------------- column aliases for ERS ----------------
ALIASES = {
    "callsign": ["Radiokallesignal (ERS)", "Radiokallesignal", "Callsign", "call_sign", "CALLSIGN"],
    "type":     ["Meldingstype", "Type", "MsgType"],
    "stop_ts":  ["Stopptidspunkt", "Tidspunkt", "Timestamp", "Tid"],
    "arrive_ts":["Ankomsttidspunkt", "Ankomst", "ETA"],
    "depart_ts":["Avgangstidspunkt", "Avgang", "ETD"],
    "gear":     ["Redskap FAO", "Redskap FDIR", "Redskap", "Gear"],
    # static
    "length_m": [
        "Største lengde",   # vessel length in all three files
        "Fartøylengde",     # also vessel length
        "length_m",         # fallback if you ever add a cleaned column
    ],
    "width_m": [
        "Bredde",           # vessel width in all three files
        "width_m",
    ],
    "draught_m":     ["Dypgående", "Dypgaende", "draught_m"],
    "engine_kw":     ["Motorkraft", "engine_kw"],
    "gross_tonnage": ["Bruttotonnasje 1969", "Bruttotonnasje annen", "GT", "gross_tonnage"],
    "vessel_type":   ["Fartøytype", "VesselType", "vessel_type"],
    "flag":          ["Fartøynasjonalitet (kode)", "Flag", "flag"],
    # catches (DCA)
    "art_fao_code":  ["Art FAO (kode)"],
    "art_fao":       ["Art FAO"],
    "art_fdir_code": ["Art - FDIR (kode)", "Art - FDIR (Kode)"],
    "art_fdir":      ["Art - FDIR"],
    "main_art_fao":  ["Hovedart FAO", "Hovedart FAO (kode)", "Hovedart - FDIR"],
    "rundvekt":      ["Rundvekt", "Rundvekt (kg)", "Rund vekt"],
}

GEAR_REPORT_DELAY_MIN = 60  # conservative to avoid leakage

# -------- utils
def pick_col(df: pd.DataFrame, names):
    if df is None:
        return None
    for n in names:
        if n in df.columns:
            return n
    return None

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def interpolate(t: int, track: np.ndarray):
    LAT, LON, SOG, COG, TS, MMSI = range(6)
    before_p = np.nonzero(t >= track[:,TS])[0]
    after_p  = np.nonzero(t <  track[:,TS])[0]
    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]
        bpos = before_p[-1]
        dt_full = float(track[apos,TS] - track[bpos,TS])
        if abs(dt_full) > 2*3600:
            return None
        dt_interp = float(t - track[bpos,TS])
        try:
            az, _, dist = geod.inv(track[bpos,1], track[bpos,0], track[apos,1], track[apos,0])
            lon_i, lat_i, _ = geod.fwd(track[bpos,1], track[bpos,0], az, dist*(dt_interp/dt_full))
            sog_i  = (track[apos,2]-track[bpos,2])*(dt_interp/dt_full) + track[bpos,2]
            cog_i  = (track[apos,3]-track[bpos,3])*(dt_interp/dt_full) + track[bpos,3]
        except Exception:
            return None
        return np.array([lat_i, lon_i, sog_i, cog_i, t, track[0,5]])
    return None

def downsample(arr: np.array, minutes: int):
    TS = 4
    sampling_track = np.empty((0, 6))
    for t in range(int(arr[0, TS]), int(arr[-1, TS]), minutes*60):
        interpolated = interpolate(t, arr)
        if interpolated is None:
            return None
        sampling_track = np.vstack([sampling_track, interpolated])
    return sampling_track

def filter_outlier_messages(arr: np.ndarray) -> np.ndarray:
    TS = 4
    q = len(arr) // 4
    first_cut = 0
    last_cut = len(arr)
    for i in range(1, len(arr)):
        if arr[i][TS] - arr[i-1][TS] > 2*3600:
            if i <= q:
                first_cut = i
            elif i >= 3*q:
                last_cut = i
            else:
                return np.array([])
    return arr[first_cut:last_cut]

def most_common(seq):
    seq = [s for s in seq if pd.notnull(s)]
    return Counter(seq).most_common(1)[0][0] if seq else None

# -------- gear timeline
def build_gear_timeline_any(ers_sources, callsign, t_start, t_end, delay_min=GEAR_REPORT_DELAY_MIN):
    cs = str(callsign).strip().upper()
    for df in ers_sources:
        if df is None or df.empty:
            continue
        gear_col = pick_col(df, ALIASES["gear"])
        if not gear_col:
            continue
        call_col = pick_col(df, ALIASES["callsign"])
        time_col = (pick_col(df, ALIASES["stop_ts"]) or
                    pick_col(df, ALIASES["arrive_ts"]) or
                    pick_col(df, ALIASES["depart_ts"]))
        if not call_col or not time_col:
            continue
        sub = df.loc[
            (df[call_col].astype(str).str.strip().str.upper() == cs) &
            (df[time_col] >= t_start) &
            (df[time_col] <= t_end) &
            (df[gear_col].notna())
        ].copy()
        if sub.empty:
            continue
        sub["ts_eff"] = pd.to_datetime(sub[time_col])
        stop_col = pick_col(df, ALIASES["stop_ts"])
        if stop_col and time_col == stop_col:
            sub["ts_eff"] = sub["ts_eff"] + pd.to_timedelta(delay_min, unit="m")
        sub = sub.sort_values("ts_eff")
        timeline, last = [], None
        for _, r in sub.iterrows():
            g = str(r[gear_col]).strip()
            ts = r["ts_eff"].to_pydatetime()
            if not timeline or g != last:
                timeline.append((ts, g))
                last = g
        if timeline:
            return timeline
    return []

def stamp_gear_on_grid(grid_ts, timeline):
    if not grid_ts:
        return [], [], []
    change_ts = [t for t,_ in timeline]
    change_g  = [g for _,g in timeline]
    gear_id=[]; changed=[]; t_since=[]; last_idx=-1; last_change_time=None
    for ts in grid_ts:
        idx = bisect_right(change_ts, ts) - 1
        if idx >= 0:
            g = change_g[idx]
            gear_id.append(g)
            if idx != last_idx:
                changed.append(True)
                last_idx = idx
                last_change_time = max(change_ts[idx], ts)
            else:
                changed.append(False)
            t_since.append(None if last_change_time is None else int((ts-last_change_time).total_seconds()//60))
        else:
            gear_id.append(None)
            changed.append(False)
            t_since.append(None)
    return gear_id, changed, t_since

# -------- DB helpers
def connect_env():
    if not PSYCOPG_AVAILABLE:
        raise ImportError("psycopg is not installed. Install with: pip install psycopg[binary]")
    return connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        dbname=os.getenv("PGDATABASE", "postgres"),
    )

def upsert_vessel(cur, mmsi, callsign, static_info):
    cur.execute("""
        INSERT INTO vessels (mmsi, callsign, type, flag, length_m, width_m, draught_m, engine_kw, gross_tonnage)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (mmsi) DO UPDATE SET
          callsign = EXCLUDED.callsign,
          type = COALESCE(EXCLUDED.type, vessels.type),
          flag = COALESCE(EXCLUDED.flag, vessels.flag),
          length_m = COALESCE(EXCLUDED.length_m, vessels.length_m),
          width_m = COALESCE(EXCLUDED.width_m, vessels.width_m),
          draught_m = COALESCE(EXCLUDED.draught_m, vessels.draught_m),
          engine_kw = COALESCE(EXCLUDED.engine_kw, vessels.engine_kw),
          gross_tonnage = COALESCE(EXCLUDED.gross_tonnage, vessels.gross_tonnage)
    """, (
        int(mmsi), callsign,
        static_info.get("vessel_type"),
        static_info.get("flag"),
        static_info.get("length_m"),
        static_info.get("width_m"),
        static_info.get("draught_m"),
        static_info.get("engine_kw"),
        static_info.get("gross_tonnage"),
    ))

def load_ports_df(con) -> pd.DataFrame:
    with con.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT port_id, name, lat, lon FROM ports")
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["port_id","name","lat","lon"])

def load_ports_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load ports from CSV.

    Prefers old behaviour:
      - 'coords' column with JSON "[lat, lon]" -> lat/lon extracted.

    Falls back to wide formats with explicit lat/lon columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ports CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols_lower = [c.lower().strip() for c in df.columns]

    # Old behaviour: single "coords" column
    if "coords" in cols_lower:
        import ast
        coords_col = df.columns[cols_lower.index("coords")]

        lats, lons = [], []
        for v in df[coords_col]:
            if pd.isna(v):
                lats.append(np.nan)
                lons.append(np.nan)
                continue
            obj = None
            try:
                obj = json.loads(str(v))
            except Exception:
                try:
                    obj = ast.literal_eval(str(v))
                except Exception:
                    obj = None
            if isinstance(obj, (list, tuple)) and len(obj) == 2:
                try:
                    lat = float(obj[0])
                    lon = float(obj[1])
                except Exception:
                    lat = lon = np.nan
            else:
                lat = lon = np.nan
            lats.append(lat)
            lons.append(lon)

        out = pd.DataFrame({"lat": lats, "lon": lons})
        out = out.dropna(subset=["lat", "lon"])
        out = out[(out["lat"] != 0) | (out["lon"] != 0)]
        if out.empty:
            raise ValueError("Parsed no valid coordinates from 'coords' column.")
        out["port_id"] = np.arange(1, len(out) + 1, dtype=int)
        return out[["port_id", "lat", "lon"]]

    # Fallback: wide formats
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    column_mapping = {}
    if 'port_id' not in df.columns:
        for col in ['portid', 'id', 'port_number']:
            if col in df.columns:
                column_mapping[col] = 'port_id'
                break
    if 'name' not in df.columns:
        for col in ['port_name', 'portname']:
            if col in df.columns:
                column_mapping[col] = 'name'
                break
    if 'lat' not in df.columns:
        for col in ['latitude', 'lat_dd', 'y']:
            if col in df.columns:
                column_mapping[col] = 'lat'
                break
    if 'lon' not in df.columns:
        for col in ['longitude', 'long', 'lon_dd', 'x']:
            if col in df.columns:
                column_mapping[col] = 'lon'
                break
    if column_mapping:
        df = df.rename(columns=column_mapping)

    required = ['port_id', 'lat', 'lon']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ports CSV missing required columns: {missing}. Found: {df.columns.tolist()}")

    before = len(df)
    df = df.dropna(subset=['lat', 'lon'])
    df = df[(df['lat'] != 0) | (df['lon'] != 0)]
    after = len(df)
    if before > after:
        print(f"[!] Filtered out {before - after} ports with missing/invalid coordinates")

    df['port_id'] = df['port_id'].astype(int)
    return df[['port_id', 'lat', 'lon']]

def nearest_port_id(lat, lon, ports_df):
    if ports_df.empty:
        return None, None
    dists = haversine_km(lon, lat, ports_df["lon"].values, ports_df["lat"].values)
    idx = int(np.argmin(dists))
    return int(ports_df.iloc[idx]["port_id"]), float(dists[idx])

# -------- ERS loading
def load_ers_files(ers_dir, year):
    dca_df = None
    por_df = None
    dep_df = None
    if not os.path.exists(ers_dir):
        return dca_df, por_df, dep_df
    for filename in os.listdir(ers_dir):
        if not filename.endswith('.csv'):
            continue
        if 'overforingsmelding' in filename.lower() or 'tra' in filename.lower():
            continue
        filepath = os.path.join(ers_dir, filename)
        if 'fangstmelding' in filename.lower() or 'dca' in filename.lower():
            dca_df = pd.read_csv(filepath, delimiter=";", low_memory=False)
        elif 'ankomstmelding' in filename.lower() or 'por' in filename.lower():
            por_df = pd.read_csv(filepath, delimiter=";", low_memory=False)
        elif 'avgangsmelding' in filename.lower() or 'dep' in filename.lower():
            dep_df = pd.read_csv(filepath, delimiter=";", low_memory=False)
    # Parse timestamps
    for df, key in [(dca_df, "stop_ts"), (por_df, "arrive_ts"), (dep_df, "depart_ts")]:
        if df is not None:
            col = pick_col(df, ALIASES[key])
            if col:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return dca_df, por_df, dep_df

# -------- static info extraction
def _parse_number(value):
    """
    Robust numeric parser for static vessel fields.
    Handles:
      - comma decimals: '53,2' -> 53.2
      - stray units/text: '55 m', '5 200 hk' -> 55, 5200
    Returns float or np.nan.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if not s:
        return np.nan

    # replace comma with dot for decimals
    s = s.replace(",", ".")

    # keep only characters that look like part of a number
    # (digits, sign, dot, exponent markers)
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    if not s:
        return np.nan

    return pd.to_numeric(s, errors="coerce")


def get_static_from_ers(ers_sources, callsign):
    static_info = {
        "length_m": None,
        "width_m": None,
        "draught_m": None,
        "engine_kw": None,
        "gross_tonnage": None,
        "vessel_type": None,
        "flag": None,
    }
    cs = str(callsign).strip().upper()

    for df in ers_sources:
        if df is None or df.empty:
            continue

        call_col = pick_col(df, ALIASES["callsign"])
        if not call_col:
            continue

        sub = df[df[call_col].astype(str).str.strip().str.upper() == cs]
        if sub.empty:
            continue

        for key in list(static_info.keys()):
            # don't overwrite once we have a value
            if static_info[key] is not None:
                continue

            col = pick_col(sub, ALIASES[key])
            if not col:
                continue

            for v in sub[col].dropna().iloc[::-1]:
                if key in ("vessel_type", "flag"):
                    val = str(v).strip()
                    if val:
                        static_info[key] = val
                        break
                else:
                    val = _parse_number(v)
                    if pd.notna(val):
                        static_info[key] = float(val)
                        break

    return static_info

# -------- catches from DCA within a voyage window
def extract_voyage_catches(dca_df: pd.DataFrame, callsign: str, t_start: datetime, t_end: datetime) -> pd.DataFrame:
    """Return aggregated catches per species for callsign within [t_start, t_end] using DCA."""
    if dca_df is None or dca_df.empty:
        return pd.DataFrame(columns=[
            "art_fao_code","art_fao","art_fdir_code","art_fdir","total_rundvekt_kg"
        ])
    call_col = pick_col(dca_df, ALIASES["callsign"])
    stop_col = pick_col(dca_df, ALIASES["stop_ts"])
    fao_code_col = pick_col(dca_df, ALIASES["art_fao_code"])
    fao_name_col = pick_col(dca_df, ALIASES["art_fao"])
    fdir_code_col = pick_col(dca_df, ALIASES["art_fdir_code"])
    fdir_name_col = pick_col(dca_df, ALIASES["art_fdir"])
    weight_col = pick_col(dca_df, ALIASES["rundvekt"])
    if not (call_col and stop_col and weight_col):
        return pd.DataFrame(columns=[
            "art_fao_code","art_fao","art_fdir_code","art_fdir","total_rundvekt_kg"
        ])

    cs = str(callsign).strip().upper()
    sub = dca_df.loc[
        (dca_df[call_col].astype(str).str.strip().str.upper() == cs) &
        (dca_df[stop_col] >= t_start) &
        (dca_df[stop_col] <= t_end)
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["art_fao_code","art_fao","art_fdir_code","art_fdir","total_rundvekt_kg"])

    # Coerce columns
    for c in [fao_code_col, fao_name_col, fdir_code_col, fdir_name_col]:
        if c and c in sub.columns:
            sub[c] = sub[c].astype(str).str.strip()
    w = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0.0)

    # Grouping keys (prefer FAO; include FDIR too when present)
    group_keys = []
    if fao_code_col:
        group_keys.append(fao_code_col)
    if fao_name_col:
        group_keys.append(fao_name_col)
    if fdir_code_col:
        group_keys.append(fdir_code_col)
    if fdir_name_col:
        group_keys.append(fdir_name_col)
    if not group_keys:
        sub = sub.assign(_w=w)
        tot = float(sub["_w"].sum())
        return pd.DataFrame([{
            "art_fao_code": None, "art_fao": None,
            "art_fdir_code": None, "art_fdir": None,
            "total_rundvekt_kg": tot
        }])

    agg = (sub.assign(_w=w)
              .groupby(group_keys, dropna=False)["_w"]
              .sum()
              .reset_index(name="total_rundvekt_kg"))

    out = pd.DataFrame({
        "art_fao_code": agg.get(fao_code_col, pd.Series([None]*len(agg))),
        "art_fao": agg.get(fao_name_col, pd.Series([None]*len(agg))),
        "art_fdir_code": agg.get(fdir_code_col, pd.Series([None]*len(agg))),
        "art_fdir": agg.get(fdir_name_col, pd.Series([None]*len(agg))),
        "total_rundvekt_kg": agg["total_rundvekt_kg"]
    })
    return out

# -------- POR-after-label counter
def count_por_after_label(por_df: pd.DataFrame, callsign: str, label_ts: datetime) -> int:
    if por_df is None or por_df.empty or label_ts is None:
        return 0
    call_col = pick_col(por_df, ALIASES["callsign"])
    arr_col  = pick_col(por_df, ALIASES["arrive_ts"])
    if not call_col or not arr_col:
        return 0
    cs = str(callsign).strip().upper()
    sub = por_df.loc[
        (por_df[call_col].astype(str).str.strip().str.upper() == cs) &
        (por_df[arr_col] > label_ts)
    ]
    return int(len(sub))

# -------- legacy-style port helper (matches old script)
def close_to_port_legacy(arr: np.ndarray, ports_df: pd.DataFrame):
    """
    Old-script style port matching:
      - ports_df must have a 'coords' column with JSON like "[lat, lon]".
      - Uses the last AIS message in `arr`.
    Returns: (in_port_flag, distance_km, (port_lat, port_lon), row_index)
    """
    last_msg = arr[-1]
    lat = float(last_msg[0])
    lon = float(last_msg[1])

    closest_dist = float("inf")
    closest_port = None
    closest_idx = None

    for idx, row in ports_df.reset_index(drop=True).iterrows():
        try:
            port_lat, port_lon = tuple(json.loads(row["coords"]))
        except Exception:
            continue
        dist = haversine_km(lon, lat, port_lon, port_lat)
        if dist < closest_dist:
            closest_dist = dist
            closest_port = (port_lat, port_lon)
            closest_idx = idx

    if closest_port is not None and closest_dist < 5.0:
        return True, closest_dist, closest_port, closest_idx
    return False, None, None, None

# -------- legacy-style AIS–ERS matching (core selection like old script)
def generate_voyages_like_old_script(
    ais_files,
    ers_csv_path,
    radio2mmsi_path,
    ports_csv_path,
    limit=None,
):
    """
    Reproduce the old script's AIS–ERS matching behaviour as closely as possible.
    Returns:
      - tracks: list of dicts with normalized traj + metadata
      - counters: dict with instrumentation counts
    """
    print("[*] Loading datasets (legacy-compatible)")

    # --- AIS (exactly like old script) ---
    ais_dfs = []
    for i, ais_file in enumerate(ais_files, 1):
        df = pd.read_csv(ais_file, delimiter=",", low_memory=False)
        ais_dfs.append(df)
        print(f"[+] Loaded AIS file {i}/{len(ais_files)}: {ais_file}")
    ais_df = pd.concat(ais_dfs, ignore_index=True)
    print(f"[+] Combined AIS Dataset: {len(ais_df)} messages")

    # Old script used strict format
    ais_df["date"] = pd.to_datetime(
        ais_df["date"],
        format="%Y-%m-%dT%H:%M:%S",
        errors="coerce",
    )

    # --- ERS merged.csv (legacy-style) ---
    if not os.path.exists(ers_csv_path):
        raise FileNotFoundError(f"ERS merged file not found: {ers_csv_path}")
    ers_df = pd.read_csv(ers_csv_path, delimiter=";", low_memory=False)
    print("[+] Loaded ERS merged.csv")

    # Parse timestamps like old script
    for col_name in ["Stopptidspunkt", "Ankomsttidspunkt", "Avgangstidspunkt"]:
        if col_name in ers_df.columns:
            ers_df[col_name] = pd.to_datetime(
                ers_df[col_name],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )

    if limit is not None:
        print(f"[!] LIMIT: Processing only first {limit} ERS rows (legacy loop)")
        ers_df = ers_df.head(limit)

    # --- Radio2MMSI & ports (old-style) ---
    radio2mmsi = pd.read_csv(
        radio2mmsi_path,
        skiprows=1,
        delimiter=";",
        index_col=0,
    ).squeeze().to_dict()
    print("[+] Loaded Radio2MMSI")

    ports_raw = pd.read_csv(ports_csv_path, delimiter=",", low_memory=False)
    if "coords" not in ports_raw.columns:
        raise ValueError("Expected 'coords' column in ports.csv to match old script.")
    print("[+] Loaded Ports (raw coords)")

    # --- Instrumentation counters ---
    counters = {
        "n_total": len(ers_df),
        "n_window_ok": 0,
        "n_has_mmsi": 0,
        "n_has_ais": 0,
        "n_quality_ok": 0,
        "n_downsample_ok": 0,
        "n_port_ok": 0,
    }

    tracks = []

    # --- Core legacy loop ---
    for _, row in tqdm(ers_df.iterrows(), desc="Collecting AIS tracks (legacy)", total=len(ers_df.index)):
        starttime = row["Stopptidspunkt"]
        arr_time  = row["Ankomsttidspunkt"]
        dep_time  = row["Avgangstidspunkt"]

        # Require all three timestamps and dep >= arr
        if pd.isna(starttime) or pd.isna(arr_time) or pd.isna(dep_time) or dep_time < arr_time:
            continue

        stoptime = arr_time + (dep_time - arr_time) / 2.0
        counters["n_window_ok"] += 1

        # Callsign / MMSI mapping: use exact old column name if present
        if "Radiokallesignal (ERS)" in ers_df.columns:
            callsign = row["Radiokallesignal (ERS)"]
        else:
            call_col = pick_col(ers_df, ALIASES["callsign"])
            callsign = row[call_col] if call_col else None

        if pd.isna(callsign):
            continue

        mmsi = radio2mmsi.get(callsign, 0)
        if not mmsi:
            continue
        counters["n_has_mmsi"] += 1

        # AIS in window (exactly like old script)
        df_ais = ais_df.loc[ais_df["mmsi"] == mmsi].copy()
        df_ais = df_ais.loc[(df_ais["date"] > starttime) & (df_ais["date"] < stoptime)]
        if df_ais.empty:
            continue
        counters["n_has_ais"] += 1

        df_ais["timestamp"] = df_ais["date"].astype(np.int64) // 10**9
        df_ais = df_ais.loc[:, ["lat", "long", "sog", "cog", "timestamp", "mmsi"]]
        arr = df_ais.to_numpy()

        # Gap filtering
        arr = filter_outlier_messages(arr)
        if len(arr) == 0:
            continue

        # Length & duration filters (20 messages, 4 hours)
        duration = datetime.fromtimestamp(arr[-1][4]) - datetime.fromtimestamp(arr[0][4])
        if len(arr) < 20 or duration < timedelta(hours=4):
            continue
        counters["n_quality_ok"] += 1

        # Save last RAW AIS point
        last_raw_lat = float(arr[-1][0])
        last_raw_lon = float(arr[-1][1])
        last_raw_ts  = int(arr[-1][4])

        # Port filter (old behaviour, 5 km, using coords JSON)
        in_port, dist2port, port_coords, port_idx = close_to_port_legacy(arr, ports_raw)
        if not in_port:
            continue
        counters["n_port_ok"] += 1

        # Downsample after port filter (like old script)
        arr_ds = downsample(arr, minutes=5)
        if arr_ds is None or len(arr_ds) == 0:
            continue
        counters["n_downsample_ok"] += 1

        # Normalize (min-max) exactly as before
        arr_ds[:, 0] = (arr_ds[:, 0].astype(np.float32) - min_lat) / (max_lat - min_lat)
        arr_ds[:, 1] = (arr_ds[:, 1].astype(np.float32) - min_lon) / (max_lon - min_lon)
        arr_ds[:, 2] = (arr_ds[:, 2].astype(np.float32) - min_sog) / (max_sog - min_sog)
        arr_ds[:, 3] = (arr_ds[:, 3].astype(np.float32) - min_cog) / (max_cog - min_cog)
        arr_ds[:, :4][arr_ds[:, :4] >= 1] = 0.9999

        label_ts = datetime.fromtimestamp(last_raw_ts)

        tracks.append({
            "mmsi": int(mmsi),
            "callsign": callsign,
            "start_time": starttime.to_pydatetime(),
            "end_time": stoptime.to_pydatetime(),
            "label_ts": label_ts,
            "label_lat": last_raw_lat,
            "label_lon": last_raw_lon,
            "port_distance_km": float(dist2port),
            "port_lat": float(port_coords[0]),
            "port_lon": float(port_coords[1]),
            "port_idx": int(port_idx),
            "traj": arr_ds,
        })

    return tracks, counters

# -------- main processing (now built on legacy-style selection)
def match_and_insert(ais_files, ers_dir, year,
                     radio2mmsi_path="data/radio2mmsi.csv",
                     output_mode="csv", csv_output_dir="output",
                     ports_csv_path="data/ports.csv", limit=None):
    print("[*] Loading datasets")

    # ERS merged.csv path (same as old script ers argument)
    year_dir = os.path.dirname(ais_files[0]) if ais_files else None
    if not year_dir:
        raise ValueError("Could not determine year_dir from AIS files")
    ers_csv_path = os.path.join(year_dir, "merged.csv")
    if not os.path.exists(ers_csv_path):
        raise FileNotFoundError(f"ERS merged file not found: {ers_csv_path}")

    # Always load original ERS files for static/gear + catches + POR counting
    dca_static, por_static, dep_static = load_ers_files(ers_dir, year)

    # Use legacy-style voyage generation
    tracks, counters = generate_voyages_like_old_script(
        ais_files=ais_files,
        ers_csv_path=ers_csv_path,
        radio2mmsi_path=radio2mmsi_path,
        ports_csv_path=ports_csv_path,
        limit=limit,
    )

    static_cache = {}

    if output_mode == "csv":
        os.makedirs(csv_output_dir, exist_ok=True)
        vessels_list, voyages_list, gear_events_list, ais_points_list, catches_list = [], [], [], [], []
        print(f"[*] CSV output mode - will save to {csv_output_dir}/")
    else:
        if not PSYCOPG_AVAILABLE:
            raise ImportError("Database mode requires psycopg or use --output csv")
        con = connect_env()
        ports_df_db = load_ports_df(con)
        vessels_list = voyages_list = gear_events_list = ais_points_list = catches_list = None  # placeholder

    # Simple port_id scheme: 1 + port_idx from legacy matching
    for tr in tqdm(tracks, desc="Enriching voyages (static/gear/catches)"):
        mmsi = tr["mmsi"]
        callsign = tr["callsign"]
        start_time = tr["start_time"]
        end_time = tr["end_time"]
        label_ts = tr["label_ts"]
        label_lat = tr["label_lat"]
        label_lon = tr["label_lon"]
        port_distance = tr["port_distance_km"]
        port_id = tr["port_idx"] + 1
        arr_ds = tr["traj"]

        # Static info
        if callsign not in static_cache:
            static_cache[callsign] = get_static_from_ers(
                [por_static, dep_static, dca_static],
                callsign
            )
        static_info = static_cache[callsign]

        # Gear timeline + summary
        timeline = build_gear_timeline_any(
            [dca_static, por_static, dep_static],
            callsign,
            start_time,
            end_time,
        )
        gear_primary = most_common([g for _, g in timeline]) if timeline else None
        n_changes = max(0, len(timeline) - 1)

        # POR arrivals after label_ts
        por_after_label_count = count_por_after_label(por_static, callsign, label_ts)

        # Catches from DCA within window
        catches_df = extract_voyage_catches(dca_static, callsign, start_time, end_time)
        total_catch_kg = float(catches_df["total_rundvekt_kg"].sum()) if not catches_df.empty else 0.0
        primary_species = None
        if not catches_df.empty:
            top = catches_df.sort_values("total_rundvekt_kg", ascending=False).iloc[0]
            primary_species = (
                top.get("art_fao")
                or top.get("art_fdir")
                or top.get("art_fao_code")
                or top.get("art_fdir_code")
            )

        voyage_id = str(uuid.uuid4())

        if output_mode == "csv":
            # vessels
            vessel_row = {
                "mmsi": int(mmsi),
                "callsign": callsign,
                "vessel_type": static_info.get("vessel_type"),
                "flag": static_info.get("flag"),
                "length_m": static_info.get("length_m"),
                "width_m": static_info.get("width_m"),
                "draught_m": static_info.get("draught_m"),
                "engine_kw": static_info.get("engine_kw"),
                "gross_tonnage": static_info.get("gross_tonnage"),
            }
            if not any(v["mmsi"] == int(mmsi) for v in vessels_list):
                vessels_list.append(vessel_row)

            # voyages
            voyages_list.append({
                "voyage_id": voyage_id,
                "mmsi": int(mmsi),
                "callsign": callsign,
                "start_ts": start_time,
                "end_ts": end_time,
                "label_port_id": port_id,
                "label_ts": label_ts,
                "label_lat": label_lat,
                "label_lon": label_lon,
                "port_distance_km": port_distance,
                "gear_primary": gear_primary,
                "n_changes": n_changes,
                "n_ais_points": len(arr_ds),
                "por_after_label_count": por_after_label_count,
                "total_catch_kg": total_catch_kg,
                "primary_species": primary_species,
            })

            # gear_events
            for t, g in timeline:
                gear_events_list.append({
                    "voyage_id": voyage_id,
                    "ts": t,
                    "gear_code": g,
                })

            # AIS points (denormalize lat/lon/sog/cog for CSV)
            grid_ts = [datetime.fromtimestamp(int(ts)) for ts in arr_ds[:, 4]]
            gear_id, changed, t_since = stamp_gear_on_grid(grid_ts, timeline)
            for idx in range(len(grid_ts)):
                lat = float(arr_ds[idx, 0] * (max_lat - min_lat) + min_lat)
                lon = float(arr_ds[idx, 1] * (max_lon - min_lon) + min_lon)
                sog = float(arr_ds[idx, 2] * (max_sog - min_sog) + min_sog)
                cog = float(arr_ds[idx, 3] * (max_cog - min_cog) + min_cog)
                ais_points_list.append({
                    "voyage_id": voyage_id,
                    "seq_idx": idx,
                    "ts": grid_ts[idx],
                    "lat": lat,
                    "lon": lon,
                    "sog": sog,
                    "cog": cog,
                    "heading": None,
                    "gear_id": gear_id[idx],
                    "gear_changed_flag": changed[idx],
                    "time_since_change_min": t_since[idx],
                })

            # catches table
            if not catches_df.empty:
                for _, r in catches_df.iterrows():
                    catches_list.append({
                        "voyage_id": voyage_id,
                        "callsign": callsign,
                        "art_fao_code": r.get("art_fao_code"),
                        "art_fao": r.get("art_fao"),
                        "art_fdir_code": r.get("art_fdir_code"),
                        "art_fdir": r.get("art_fdir"),
                        "total_rundvekt_kg": float(r.get("total_rundvekt_kg", 0.0)),
                    })

    # -------- Save CSVs
    if output_mode == "csv":
        pd.DataFrame(vessels_list).to_csv(os.path.join(csv_output_dir, f"vessels_{year}.csv"), index=False)
        pd.DataFrame(voyages_list).to_csv(os.path.join(csv_output_dir, f"voyages_{year}.csv"), index=False)
        if gear_events_list:
            pd.DataFrame(gear_events_list).to_csv(os.path.join(csv_output_dir, f"gear_events_{year}.csv"), index=False)
        pd.DataFrame(ais_points_list).to_csv(os.path.join(csv_output_dir, f"ais_points_{year}.csv"), index=False)
        pd.DataFrame(catches_list).to_csv(os.path.join(csv_output_dir, f"voyage_catches_{year}.csv"), index=False)
        print(f"[+] Wrote CSVs to {csv_output_dir}/ (incl. voyage_catches_{year}.csv)")
    else:
        print("[!] DB mode not extended to insert catches table in this script (export CSV instead).")

    # --- instrumentation summary (using legacy counters) ---
    print("\n[Instrumentation summary]")
    print(f"  Total ERS rows (driver_df): {counters['n_total']}")
    print(f"  With valid time window:     {counters['n_window_ok']}")
    print(f"  With MMSI match:           {counters['n_has_mmsi']}")
    print(f"  With AIS in window:        {counters['n_has_ais']}")
    print(f"  Pass quality (len/dur):    {counters['n_quality_ok']}")
    print(f"  Pass downsample:           {counters['n_downsample_ok']}")
    print(f"  Pass port filter:          {counters['n_port_ok']}")
    print(f"  Final voyages written:     {len(voyages_list) if output_mode=='csv' else 'N/A (DB mode)'}")

# -------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build voyages with AIS+ERS and attach catches from DCA; output CSVs")
    ap.add_argument("base_dir", type=str, help="Base directory (e.g., src/data)")
    ap.add_argument("year", type=str, help="Year to process (e.g., '2016')")
    ap.add_argument("--radio2mmsi", type=str, default=None, help="callsign→MMSI map (default: BASE_DIR/radio2mmsi.csv)")
    ap.add_argument("--output", type=str, choices=["csv"], default="csv", help="Output mode")
    ap.add_argument("--csv-dir", type=str, default="output", help="Directory to save CSV files")
    ap.add_argument("--ports-csv", type=str, default=None, help="Path to ports CSV (default: BASE_DIR/ports.csv)")
    ap.add_argument("--limit", type=int, default=None, help="Limit processing to first N ERS rows")
    args = ap.parse_args()

    if args.radio2mmsi is None:
        args.radio2mmsi = os.path.join(args.base_dir, "radio2mmsi.csv")
    if args.ports_csv is None:
        args.ports_csv = os.path.join(args.base_dir, "ports.csv")

    base_dir = os.path.abspath(args.base_dir)
    ais_dir = os.path.join(base_dir, args.year)
    ers_dir = os.path.join(base_dir, "ers", args.year)

    ais_files = []
    if os.path.exists(ais_dir):
        for file in os.listdir(ais_dir):
            if file.endswith('.csv') and file.lower() != 'merged.csv' and file != 'combined.csv':
                ais_files.append(os.path.join(ais_dir, file))
    if not ais_files:
        print(f"Error: No AIS files found in {ais_dir}")
        exit(1)

    if not os.path.exists(ers_dir):
        print(f"[i] ERS directory not found: {ers_dir} — static/gear/catches will be limited.")

    if not os.path.exists(args.ports_csv):
        print(f"Error: Ports CSV file not found: {args.ports_csv}")
        exit(1)

    ais_files.sort()

    print(f"\nProcessing data for year {args.year}")
    print(f"Output dir: {os.path.abspath(args.csv_dir)}")
    print(f"Ports reference file: {args.ports_csv}")
    if args.limit:
        print(f"LIMIT: Processing only first {args.limit} ERS rows")
    print(f"Found {len(ais_files)} AIS files:")
    for f in ais_files:
        print(f"  - {os.path.basename(f)}")

    match_and_insert(
        ais_files, ers_dir, args.year, args.radio2mmsi,
        output_mode=args.output, csv_output_dir=args.csv_dir,
        ports_csv_path=args.ports_csv, limit=args.limit
    )
