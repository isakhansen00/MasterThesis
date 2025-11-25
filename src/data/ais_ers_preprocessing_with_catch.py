#!/usr/bin/env python3
import os, uuid, argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from bisect import bisect_right
from pyproj import Geod
from tqdm import tqdm

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
- Voyage windows from per-year ERS merged.csv (Stopptidspunkt, Ankomsttidspunkt, Avgangstidspunkt)
- Static info + gear usage from ERS DCA/POR/DEP (ers/<year>/...)
- AIS selection/filters; enforce last AIS point ≤ PORT_MAX_KM to a known port
- NEW: voyage_catches_{year}.csv with aggregated DCA catches per voyage (by species)
"""

geod = Geod(ellps='WGS84')

# ---------------- normalization bounds (region-specific) ----------------
min_cog, max_cog = 0.0, 360.0
min_sog, max_sog = 0, 30.0
min_lat, max_lat = 69.2, 73.0
min_lon, max_lon = 13.0, 31.5

# Require end-of-window to be within N km of a port
PORT_MAX_KM = 10.0

# ---------------- column aliases for ERS ----------------
ALIASES = {
    "callsign": ["Radiokallesignal (ERS)", "Radiokallesignal", "Callsign", "call_sign", "CALLSIGN"],
    "type":     ["Meldingstype", "Type", "MsgType"],
    "stop_ts":  ["Stopptidspunkt", "Tidspunkt", "Timestamp", "Tid"],
    "arrive_ts":["Ankomsttidspunkt", "Ankomst", "ETA"],
    "depart_ts":["Avgangstidspunkt", "Avgang", "ETD"],
    "gear":     ["Redskap FAO", "Redskap FDIR", "Redskap", "Gear"],
    # static
    "length_m":      ["Største lengde", "Fartøylengde", "Lengde", "length_m"],
    "width_m":       ["Bredde", "width_m"],
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
    if df is None: return None
    for n in names:
        if n in df.columns: return n
    return None

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def interpolate(t: int, track: np.ndarray):
    LAT, LON, SOG, COG, TS, MMSI = range(6)
    before_p = np.nonzero(t >= track[:,TS])[0]
    after_p  = np.nonzero(t <  track[:,TS])[0]
    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]; bpos = before_p[-1]
        dt_full = float(track[apos,TS] - track[bpos,TS])
        if abs(dt_full) > 2*3600: return None
        dt_interp = float(t - track[bpos,TS])
        try:
            az, _, dist = geod.inv(track[bpos,1], track[bpos,0], track[apos,1], track[apos,0])
            lon_i, lat_i, _ = geod.fwd(track[bpos,1], track[bpos,0], az, dist*(dt_interp/dt_full))
            sog_i  = (track[apos,2]-track[bpos,2])*(dt_interp/dt_full) + track[bpos,2]
            cog_i  = (track[apos,3]-track[bpos,3])*(dt_interp/dt_full) + track[bpos,3]
        except:
            return None
        return np.array([lat_i, lon_i, sog_i, cog_i, t, track[0,5]])
    return None

def downsample(arr: np.array, minutes: int):
    TS = 4
    sampling_track = np.empty((0, 6))
    for t in range(int(arr[0, TS]), int(arr[-1, TS]), minutes*60):
        interpolated = interpolate(t, arr)
        if interpolated is None: return None
        sampling_track = np.vstack([sampling_track, interpolated])
    return sampling_track

def filter_outlier_messages(arr: np.ndarray) -> np.ndarray:
    TS = 4
    q = len(arr) // 4
    first_cut = 0; last_cut = len(arr)
    for i in range(1, len(arr)):
        if arr[i][TS] - arr[i-1][TS] > 2*3600:
            if i <= q: first_cut = i
            elif i >= 3*q: last_cut = i
            else: return np.array([])
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
                timeline.append((ts, g)); last = g
        if timeline:
            return timeline
    return []

def stamp_gear_on_grid(grid_ts, timeline):
    if not grid_ts: return [], [], []
    change_ts = [t for t,_ in timeline]
    change_g  = [g for _,g in timeline]
    gear_id=[]; changed=[]; t_since=[]; last_idx=-1; last_change_time=None
    for ts in grid_ts:
        idx = bisect_right(change_ts, ts) - 1
        if idx >= 0:
            g = change_g[idx]
            gear_id.append(g)
            if idx != last_idx:
                changed.append(True); last_idx = idx; last_change_time = max(change_ts[idx], ts)
            else:
                changed.append(False)
            t_since.append(None if last_change_time is None else int((ts-last_change_time).total_seconds()//60))
        else:
            gear_id.append(None); changed.append(False); t_since.append(None)
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
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ports CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    column_mapping = {}
    if 'port_id' not in df.columns:
        for col in ['portid', 'id', 'port_number']:
            if col in df.columns:
                column_mapping[col] = 'port_id'; break
    if 'name' not in df.columns:
        for col in ['port_name', 'portname']:
            if col in df.columns:
                column_mapping[col] = 'name'; break
    if 'lat' not in df.columns:
        for col in ['latitude', 'lat_dd', 'y']:
            if col in df.columns:
                column_mapping[col] = 'lat'; break
    if 'lon' not in df.columns:
        for col in ['longitude', 'long', 'lon_dd', 'x']:
            if col in df.columns:
                column_mapping[col] = 'lon'; break
    if column_mapping:
        df = df.rename(columns=column_mapping)
    required = ['port_id', 'lat', 'lon']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Ports CSV missing required columns: {missing}. Found: {df.columns.tolist()}")
    before_count = len(df)
    df = df.dropna(subset=['lat', 'lon'])
    df = df[(df['lat'] != 0) | (df['lon'] != 0)]
    after_count = len(df)
    if before_count > after_count:
        print(f"[!] Filtered out {before_count - after_count} ports with missing/invalid coordinates")
    df['port_id'] = df['port_id'].astype(int)
    return df[['port_id', 'name', 'lat', 'lon']] if 'name' in df.columns else df[['port_id', 'lat', 'lon']]

def nearest_port_id(lat, lon, ports_df):
    if ports_df.empty: return None, None
    dists = haversine_km(lon, lat, ports_df["lon"].values, ports_df["lat"].values)
    idx = int(np.argmin(dists))
    return int(ports_df.iloc[idx]["port_id"]), float(dists[idx])

# -------- ERS loading
def load_ers_files(ers_dir, year):
    dca_df = None; por_df = None; dep_df = None
    if not os.path.exists(ers_dir):
        return dca_df, por_df, dep_df
    for filename in os.listdir(ers_dir):
        if not filename.endswith('.csv'): continue
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

def load_ers_merged(ais_year_dir: str):
    merged_path = os.path.join(ais_year_dir, "merged.csv")
    if not os.path.exists(merged_path):
        return None
    print(f"[*] Loading ERS merged file: {merged_path}")
    df = pd.read_csv(merged_path, delimiter=";", low_memory=False)
    def _parse(ts):
        return pd.to_datetime(ts, errors="coerce", dayfirst=True)
    for cols in (ALIASES["stop_ts"], ALIASES["arrive_ts"], ALIASES["depart_ts"]):
        col = pick_col(df, cols)
        if col:
            df[col] = _parse(df[col])
    return df

def next_event_after(merged_df: pd.DataFrame, callsign: str, ts: pd.Timestamp):
    call_col = pick_col(merged_df, ALIASES["callsign"])
    stop_col = pick_col(merged_df, ALIASES["stop_ts"])
    arr_col  = pick_col(merged_df, ALIASES["arrive_ts"])
    dep_col  = pick_col(merged_df, ALIASES["depart_ts"])
    sub = merged_df.loc[merged_df[call_col] == callsign, [stop_col, arr_col, dep_col]].copy()
    stamps = pd.to_datetime(pd.Series(pd.concat([sub[stop_col], sub[arr_col], sub[dep_col]])), errors="coerce")
    stamps = stamps[pd.notna(stamps) & (stamps > ts)]
    return stamps.min() if not stamps.empty else None

def derive_voyage_window_from_row(row, merged_df):
    call_col = pick_col(merged_df, ALIASES["callsign"])
    stop_col = pick_col(merged_df, ALIASES["stop_ts"])
    arr_col  = pick_col(merged_df, ALIASES["arrive_ts"])
    dep_col  = pick_col(merged_df, ALIASES["depart_ts"])
    stop_time = row.get(stop_col, None)
    arr_time  = row.get(arr_col, None)
    dep_time  = row.get(dep_col, None)
    if pd.notna(stop_time) and pd.notna(arr_time) and pd.notna(dep_time) and dep_time >= arr_time:
        end_time = arr_time + (dep_time - arr_time) / 2
        return stop_time, end_time, arr_time, dep_time
    if pd.notna(stop_time) and pd.notna(arr_time):
        return stop_time, arr_time, arr_time, None
    if pd.notna(stop_time):
        next_ts = next_event_after(merged_df, row[call_col], stop_time)
        if next_ts is not None:
            return stop_time, next_ts, None, None
        return stop_time, stop_time + pd.Timedelta(days=1), None, None
    return None, None, None, None

# -------- static info extraction
def get_static_from_ers(ers_sources, callsign):
    static_info = {
        "length_m": None, "width_m": None, "draught_m": None,
        "engine_kw": None, "gross_tonnage": None, "vessel_type": None, "flag": None
    }
    cs = str(callsign).strip().upper()
    for df in ers_sources:
        if df is None or df.empty: continue
        call_col = pick_col(df, ALIASES["callsign"])
        if not call_col: continue
        sub = df[df[call_col].astype(str).str.strip().str.upper() == cs]
        if sub.empty: continue
        for key in list(static_info.keys()):
            if static_info[key] is not None: continue
            col = pick_col(sub, ALIASES[key])
            if not col: continue
            for v in sub[col].dropna().iloc[::-1]:
                if key in ("vessel_type","flag"):
                    val = str(v).strip()
                    if val:
                        static_info[key] = val; break
                else:
                    val = pd.to_numeric(v, errors="coerce")
                    if pd.notna(val):
                        static_info[key] = float(val); break
    return static_info

# -------- legacy POR/DEP matching (fallback)
def match_voyage_from_por_dep(por_df, dep_df, callsign, dca_stop_time, time_window_hours=168):
    call_col_por = pick_col(por_df, ALIASES["callsign"])
    call_col_dep = pick_col(dep_df, ALIASES["callsign"])
    arrive_col = pick_col(por_df, ALIASES["arrive_ts"])
    depart_col = pick_col(dep_df, ALIASES["depart_ts"])
    if not all([call_col_por, call_col_dep, arrive_col, depart_col]):
        return None
    arrivals = por_df.loc[
        (por_df[call_col_por] == callsign) &
        (por_df[arrive_col] > dca_stop_time) &
        (por_df[arrive_col] <= dca_stop_time + pd.Timedelta(hours=time_window_hours))
    ].sort_values(arrive_col)
    if arrivals.empty:
        return None
    arrival_time = arrivals.iloc[0][arrive_col]
    departures = dep_df.loc[
        (dep_df[call_col_dep] == callsign) &
        (dep_df[depart_col] >= arrival_time)
    ].sort_values(depart_col)
    if departures.empty:
        return None
    departure_time = departures.iloc[0][depart_col]
    return arrival_time, departure_time

# -------- AIS helpers
def load_and_concat_ais_files(ais_files):
    dfs = []
    for file in ais_files:
        if 'merged' in os.path.basename(file).lower():
            continue
        df = pd.read_csv(file, delimiter=",", low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def index_ais_by_mmsi(ais_df):
    ais_grouped = {}
    for mmsi, group in ais_df.groupby('mmsi'):
        ais_grouped[mmsi] = group.sort_values('date').copy()
    return ais_grouped

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
    if fao_code_col: group_keys.append(fao_code_col)
    if fao_name_col: group_keys.append(fao_name_col)
    if fdir_code_col: group_keys.append(fdir_code_col)
    if fdir_name_col: group_keys.append(fdir_name_col)
    if not group_keys:  # if no species columns found, aggregate as single bucket
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

    # Normalize column names in output
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

# -------- main processing
def match_and_insert(ais_files: list, ers_dir: str, year: str, radio2mmsi_path="data/radio2mmsi.csv",
                     output_mode="csv", csv_output_dir="output", ports_csv_path="data/ports.csv", limit=None):
    print("[*] Loading datasets")
    year_dir = os.path.dirname(ais_files[0]) if ais_files else None
    merged_df = load_ers_merged(year_dir) if year_dir else None

    # Always load original ERS files for static/gear + catches + POR counting
    dca_static, por_static, dep_static = load_ers_files(ers_dir, year)

    # Voyage driver
    if merged_df is not None:
        driver_df = merged_df.copy()
        print("[+] Using ERS merged.csv for voyage windows")
    else:
        driver_df = dca_static
        print("[+] Using legacy ERS DCA for voyage windows")

    ais_df = load_and_concat_ais_files(ais_files)
    radio2mmsi = pd.read_csv(radio2mmsi_path, skiprows=1, delimiter=";", index_col=0).squeeze().to_dict()
    print("[+] Datasets loaded")

    if driver_df is None:
        raise ValueError("No ERS data found - required for voyage processing")

    # AIS filter to year
    ais_df["date"] = pd.to_datetime(ais_df["date"], errors="coerce")
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{int(year)+1}-01-01")
    ais_df = ais_df[(ais_df["date"] >= year_start) & (ais_df["date"] < year_end)].copy()

    ais_by_mmsi = index_ais_by_mmsi(ais_df)

    call_col = pick_col(driver_df, ALIASES["callsign"])
    stop_col = pick_col(driver_df, ALIASES["stop_ts"])
    if call_col is None or stop_col is None:
        raise ValueError("ERS data missing required columns (callsign/Stopptidspunkt)")

    if limit is not None:
        print(f"[!] LIMIT: Processing only first {limit} ERS rows (test mode)")
        driver_df = driver_df.head(limit)

    static_cache = {}
    total_por_after_label = 0

    if output_mode == "csv":
        os.makedirs(csv_output_dir, exist_ok=True)
        vessels_list, voyages_list, gear_events_list, ais_points_list, catches_list = [], [], [], [], []
        ports_df = load_ports_from_csv(ports_csv_path)
        print(f"[*] CSV output mode - will save to {csv_output_dir}/")
        print(f"[*] Loaded {len(ports_df)} ports from {ports_csv_path}")
    else:
        if not PSYCOPG_AVAILABLE:
            raise ImportError("Database mode requires psycopg or use --output csv")
        con = connect_env()
        ports_df = load_ports_df(con)

    for _, ers_row in tqdm(driver_df.iterrows(), total=len(driver_df.index), desc="Processing voyages"):
        callsign = ers_row[call_col]
        if pd.isna(callsign):
            continue

        # Time window
        if merged_df is not None:
            start_time, end_time, arrival_time, departure_time = derive_voyage_window_from_row(ers_row, merged_df)
        else:
            stop_time = ers_row[stop_col]
            if pd.isna(stop_time): continue
            arrival_time = departure_time = None
            if por_static is not None and dep_static is not None:
                match = match_voyage_from_por_dep(por_static, dep_static, callsign, stop_time)
                if match:
                    arrival_time, departure_time = match
            if pd.notna(arrival_time) and pd.notna(departure_time):
                start_time, end_time = stop_time, arrival_time + (departure_time - arrival_time)/2
            elif pd.notna(arrival_time):
                start_time, end_time = stop_time, arrival_time
            else:
                start_time, end_time = stop_time, stop_time + pd.Timedelta(days=1)

        if start_time is None or end_time is None or end_time <= start_time:
            continue

        # MMSI mapping
        mmsi = radio2mmsi.get(callsign, 0)
        if not mmsi or mmsi not in ais_by_mmsi:
            continue

        # AIS in window
        df = ais_by_mmsi[mmsi]
        df = df.loc[(df["date"] > start_time) & (df["date"] < end_time)].copy()
        if df.empty:
            continue

        df["timestamp"] = df["date"].astype(np.int64) // 10**9
        df = df.loc[:, ["lat", "long", "sog", "cog", "timestamp", "mmsi"]]
        arr = df.to_numpy()

        # quality
        arr = filter_outlier_messages(arr)
        if len(arr) == 0:
            continue
        if len(arr) < 20 or (datetime.fromtimestamp(arr[-1][-2]) - datetime.fromtimestamp(arr[0][-2]) < timedelta(hours=4)):
            continue

        # interpolate
        arr_ds = downsample(arr, minutes=5)
        if arr_ds is None or len(arr_ds) == 0:
            continue

        # normalize (for storage consistency)
        arr_ds[:,0] = (arr_ds[:,0].astype(np.float32) - min_lat)/(max_lat - min_lat)
        arr_ds[:,1] = (arr_ds[:,1].astype(np.float32) - min_lon)/(max_lon - min_lon)
        arr_ds[:,2] = (arr_ds[:,2].astype(np.float32) - min_sog)/(max_sog - min_sog)
        arr_ds[:,3] = (arr_ds[:,3].astype(np.float32) - min_cog)/(max_cog - min_cog)
        arr_ds[:, :4][arr_ds[:, :4] >= 1] = 0.9999

        # nearest port from last point
        last_norm = arr_ds[-1]
        lat_real = float(last_norm[0]*(max_lat-min_lat) + min_lat)
        lon_real = float(last_norm[1]*(max_lon-min_lon) + min_lon)
        label_port_id, port_distance = (None, None)
        if not ports_df.empty:
            label_port_id, port_distance = nearest_port_id(lat_real, lon_real, ports_df)
        if PORT_MAX_KM is not None and (port_distance is None or port_distance > PORT_MAX_KM):
            continue

        label_ts = datetime.fromtimestamp(int(arr_ds[-1][4]))

        # static, gear
        if callsign not in static_cache:
            static_cache[callsign] = get_static_from_ers([dca_static, por_static, dep_static], callsign)
        static_info = static_cache[callsign]

        timeline = build_gear_timeline_any([dca_static, por_static, dep_static], callsign, start_time, end_time)
        gear_primary = most_common([g for _,g in timeline]) if timeline else None
        n_changes = max(0, len(timeline)-1)

        # POR arrivals after label ts
        por_after_label_count = count_por_after_label(por_static, callsign, label_ts)
        total_por_after_label += por_after_label_count

        # CATCHES from DCA within window
        catches_df = extract_voyage_catches(dca_static, callsign, start_time, end_time)
        total_catch_kg = float(catches_df["total_rundvekt_kg"].sum()) if not catches_df.empty else 0.0
        primary_species = None
        if not catches_df.empty:
            top = catches_df.sort_values("total_rundvekt_kg", ascending=False).iloc[0]
            primary_species = (top.get("art_fao") or top.get("art_fdir") or top.get("art_fao_code") or top.get("art_fdir_code"))

        voyage_id = str(uuid.uuid4())

        # ------ OUTPUTS (CSV default)
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
            "gross_tonnage": static_info.get("gross_tonnage")
        }
        if output_mode == "csv":
            if "vessels_list" not in locals():
                vessels_list, voyages_list, gear_events_list, ais_points_list, catches_list = [], [], [], [], []
            if not any(v["mmsi"] == int(mmsi) for v in vessels_list):
                vessels_list.append(vessel_row)

            # voyages
            voyages_list.append({
                "voyage_id": voyage_id,
                "mmsi": int(mmsi),
                "callsign": callsign,
                "start_ts": start_time.to_pydatetime(),
                "end_ts": end_time.to_pydatetime(),
                "label_port_id": label_port_id,
                "label_ts": label_ts,
                "label_lat": lat_real,
                "label_lon": lon_real,
                "port_distance_km": port_distance,
                "gear_primary": gear_primary,
                "n_changes": n_changes,
                "n_ais_points": len(arr_ds),
                "por_after_label_count": por_after_label_count,
                "total_catch_kg": total_catch_kg,
                "primary_species": primary_species
            })

            # gear_events
            for t, g in timeline:
                gear_events_list.append({"voyage_id": voyage_id, "ts": t, "gear_code": g})

            # ais_points
            grid_ts = [datetime.fromtimestamp(int(ts)) for ts in arr_ds[:,4]]
            gear_id, changed, t_since = stamp_gear_on_grid(grid_ts, timeline)
            for idx in range(len(grid_ts)):
                lat = float(arr_ds[idx,0]*(max_lat-min_lat) + min_lat)
                lon = float(arr_ds[idx,1]*(max_lon-min_lon) + min_lon)
                sog = float(arr_ds[idx,2]*(max_sog-min_sog) + min_sog)
                cog = float(arr_ds[idx,3]*(max_cog-min_cog) + min_cog)
                ais_points_list.append({
                    "voyage_id": voyage_id,
                    "seq_idx": idx,
                    "ts": grid_ts[idx],
                    "lat": lat, "lon": lon, "sog": sog, "cog": cog,
                    "heading": None,
                    "gear_id": gear_id[idx],
                    "gear_changed_flag": changed[idx],
                    "time_since_change_min": t_since[idx]
                })

            # catches (one row per species)
            if not catches_df.empty:
                for _, r in catches_df.iterrows():
                    catches_list.append({
                        "voyage_id": voyage_id,
                        "callsign": callsign,
                        "art_fao_code": r.get("art_fao_code"),
                        "art_fao": r.get("art_fao"),
                        "art_fdir_code": r.get("art_fdir_code"),
                        "art_fdir": r.get("art_fdir"),
                        "total_rundvekt_kg": float(r.get("total_rundvekt_kg", 0.0))
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