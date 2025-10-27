import os, uuid, argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from bisect import bisect_right
from pyproj import Geod
from tqdm import tqdm

from psycopg import connect
from psycopg.rows import dict_row


"""
This code is borrowed and modified from Andreas Løvlands code to also include static information, the original code is found at: munin.uit.no/handle/10037/34242
in the code.zip download.

"""


geod = Geod(ellps='WGS84')

# ---------------- normalization bounds (region-specific) ----------------
min_cog, max_cog = 0.0, 360.0
min_sog, max_sog = 0, 30.0
min_lat, max_lat = 69.2, 73.0
min_lon, max_lon = 13.0, 31.5

# ---------------- column aliases ----------------
ALIASES = {
    "callsign": ["Radiokallesignal (ERS)", "Radiokallesignal", "Callsign", "call_sign", "CALLSIGN"],
    "type":     ["Type", "Meldingstype", "MsgType"],
    "stop_ts":  ["Stopptidspunkt", "Tidspunkt", "Timestamp", "Tid"],
    "arrive_ts":["Ankomsttidspunkt", "Ankomst", "ETA"],
    "depart_ts":["Avgangstidspunkt", "Avgang", "ETD"],
    "gear":     ["Redskap", "Gear", "Redskapstype"],
    # static
    "length_m":      ["Lengde", "Lengde (m)", "Fartøylengde", "length_m", "Length_m"],
    "width_m":       ["Bredde", "Bredde (m)", "width_m", "Width_m"],
    "draught_m":     ["Dypgående", "Dypgaende", "Dybde", "draught_m"],
    "engine_kw":     ["Motorkraft (kW)", "Motor (kW)", "EngineKW", "engine_kw"],
    "gross_tonnage": ["Bruttotonnasje", "GT", "GrossTonnage", "gross_tonnage"],
    "vessel_type":   ["Fartøytype", "VesselType", "vessel_type"],
    "flag":          ["Flagg", "Flag", "flag"]
}

GEAR_REPORT_DELAY_MIN = 60  # conservative to avoid leakage

def pick_col(df: pd.DataFrame, names):
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

def build_gear_timeline(ers_df, callsign, t_start, t_end, delay_min=GEAR_REPORT_DELAY_MIN):
    type_col = pick_col(ers_df, ALIASES["type"])
    gear_col = pick_col(ers_df, ALIASES["gear"])
    call_col = pick_col(ers_df, ALIASES["callsign"])
    time_col = pick_col(ers_df, ALIASES["stop_ts"])
    if not all([type_col, gear_col, call_col, time_col]): return []
    df = ers_df
    if "DCA" in set(df[type_col].dropna().unique()):
        df = df.loc[df[type_col] == "DCA"]
    df = df.loc[
        (df[call_col] == callsign) &
        (df[time_col] >= t_start) &
        (df[time_col] <= t_end) &
        (df[gear_col].notna())
    ].copy()
    if df.empty: return []
    df["ts_eff"] = pd.to_datetime(df[time_col]) + pd.to_timedelta(delay_min, unit="m")
    df = df.sort_values("ts_eff")
    timeline = []
    last = None
    for _, r in df.iterrows():
        g = str(r[gear_col]).strip()
        ts = r["ts_eff"].to_pydatetime()
        if not timeline or g != last:
            timeline.append((ts, g)); last = g
    return timeline

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

# ---------------- DB helpers (psycopg3) ----------------
def connect_env():
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

def nearest_port_id(lat, lon, ports_df):
    if ports_df.empty: return None, None
    dists = haversine_km(lon, lat, ports_df["lon"].values, ports_df["lat"].values)
    idx = int(np.argmin(dists))
    return int(ports_df.iloc[idx]["port_id"]), float(dists[idx])

# ---------------- ERS static extract ----------------
def get_static_from_ers(ers_df, callsign):
    call_col = pick_col(ers_df, ALIASES["callsign"])
    ts_cols  = [c for c in [pick_col(ers_df, ALIASES["stop_ts"]),
                            pick_col(ers_df, ALIASES["arrive_ts"]),
                            pick_col(ers_df, ALIASES["depart_ts"])] if c]
    if call_col is None or not ts_cols:
        return {k: None for k in ["length_m","width_m","draught_m","engine_kw","gross_tonnage","vessel_type","flag"]}
    df = ers_df.loc[ers_df[call_col] == callsign].copy()
    df["__ts__"] = pd.to_datetime(df[ts_cols].bfill(axis=1).iloc[:,0], errors="coerce")
    df = df.sort_values("__ts__", ascending=False)

    def pull(alias_key, cast=float):
        col = pick_col(df, ALIASES[alias_key])
        if col is None: return None
        for v in df[col]:
            if pd.notnull(v):
                try: return cast(v) if cast is not str else str(v)
                except: return None
        return None

    return {
        "length_m":      pull("length_m", float),
        "width_m":       pull("width_m", float),
        "draught_m":     pull("draught_m", float),
        "engine_kw":     pull("engine_kw", float),
        "gross_tonnage": pull("gross_tonnage", float),
        "vessel_type":   pull("vessel_type", str),
        "flag":          pull("flag", str),
    }

# ---------------- main ----------------
def load_and_concat_ais_files(ais_files):
    """Load and concatenate multiple AIS files into a single DataFrame."""
    dfs = []
    for file in ais_files:
        print(f"[*] Loading AIS file: {os.path.basename(file)}")
        df = pd.read_csv(file, delimiter=",")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def match_and_insert(ais_files: list, ers_filename: str, radio2mmsi_path="data/radio2mmsi.csv"):
    print("[*] Loading datasets")
    ais_df = load_and_concat_ais_files(ais_files)
    ers_df = pd.read_csv(ers_filename, delimiter=";")
    radio2mmsi = pd.read_csv(radio2mmsi_path, skiprows=1, delimiter=";", index_col=0).squeeze().to_dict()
    print("[+] Datasets loaded")

    ais_df["date"] = pd.to_datetime(ais_df["date"], errors="coerce")
    for key in ["stop_ts","arrive_ts","depart_ts"]:
        col = pick_col(ers_df, ALIASES[key])
        if col:
            ers_df[col] = pd.to_datetime(ers_df[col], errors="coerce")

    call_col   = pick_col(ers_df, ALIASES["callsign"])
    stop_col   = pick_col(ers_df, ALIASES["stop_ts"])
    arrive_col = pick_col(ers_df, ALIASES["arrive_ts"])
    depart_col = pick_col(ers_df, ALIASES["depart_ts"])
    if call_col is None or arrive_col is None or depart_col is None or stop_col is None:
        raise ValueError("ERS file missing required columns. Update ALIASES mapping.")

    static_cache = {}

    with connect_env() as con:
        ports_df = load_ports_df(con)

        for _, row in tqdm(ers_df.iterrows(), total=len(ers_df.index), desc="Processing voyages"):
            starttime = row[stop_col]
            ankomst   = row[arrive_col]
            avgang    = row[depart_col]
            if pd.isna(starttime) or pd.isna(ankomst) or pd.isna(avgang):
                continue

            end_for_encoder = ankomst + (avgang - ankomst) / 2
            callsign = row[call_col]
            mmsi     = radio2mmsi.get(callsign, 0)
            if not mmsi:
                continue

            df = ais_df.loc[(ais_df["mmsi"] == mmsi) &
                            (ais_df["date"] > starttime) &
                            (ais_df["date"] < end_for_encoder)].copy()
            if df.empty:
                continue
            df["timestamp"] = df["date"].astype(np.int64) // 10**9
            df = df.loc[:, ["lat", "long", "sog", "cog", "timestamp", "mmsi"]]
            arr = df.to_numpy()

            arr = filter_outlier_messages(arr)
            if len(arr) == 0:
                continue
            if len(arr) < 20 or (datetime.fromtimestamp(arr[-1][-2]) - datetime.fromtimestamp(arr[0][-2]) < timedelta(hours=4)):
                continue

            arr_ds = downsample(arr, minutes=5)
            if arr_ds is None or len(arr_ds) == 0:
                continue

            arr_ds[:,0] = (arr_ds[:,0].astype(np.float32) - min_lat)/(max_lat - min_lat)
            arr_ds[:,1] = (arr_ds[:,1].astype(np.float32) - min_lon)/(max_lon - min_lon)
            arr_ds[:,2] = (arr_ds[:,2].astype(np.float32) - min_sog)/(max_sog - min_sog)
            arr_ds[:,3] = (arr_ds[:,3].astype(np.float32) - min_cog)/(max_cog - min_cog)
            arr_ds[:, :4][arr_ds[:, :4] >= 1] = 0.9999

            timeline = build_gear_timeline(ers_df, callsign, starttime, end_for_encoder, delay_min=GEAR_REPORT_DELAY_MIN)
            gear_primary = most_common([g for _,g in timeline]) if timeline else None
            n_changes = max(0, len(timeline)-1)

            last_norm = arr_ds[-1]
            lat_real = float(last_norm[0]*(max_lat-min_lat) + min_lat)
            lon_real = float(last_norm[1]*(max_lon-min_lon) + min_lon)
            label_port_id, _ = nearest_port_id(lat_real, lon_real, ports_df)
            label_ts = datetime.fromtimestamp(int(arr_ds[-1][4]))

            # Upsert vessel
            if callsign not in static_cache:
                static_cache[callsign] = get_static_from_ers(ers_df, callsign)
            static_info = static_cache[callsign]

            with con.cursor() as cur:
                upsert_vessel(cur, mmsi, callsign, static_info)

                voyage_id = uuid.uuid4()
                cur.execute("""
                    INSERT INTO voyages (voyage_id, mmsi, callsign, start_ts, end_ts,
                                        dep_port_id, arr_port_id, label_port_id, label_ts,
                                        gear_primary, n_changes)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    str(voyage_id), int(mmsi), callsign,
                    starttime.to_pydatetime(), end_for_encoder.to_pydatetime(),
                    None, None, label_port_id, label_ts, gear_primary, n_changes
                ))

                # Gear events
                if timeline:
                    cur.executemany(
                        "INSERT INTO gear_events (voyage_id, ts, gear_code) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                        [(str(voyage_id), t, g) for (t, g) in timeline],
                    )

                # AIS points (denormalized coords)
                grid_ts = [datetime.fromtimestamp(int(ts)) for ts in arr_ds[:,4]]
                gear_id, changed, t_since = stamp_gear_on_grid(grid_ts, timeline)

                rows = []
                for idx in range(len(grid_ts)):
                    lat = float(arr_ds[idx,0]*(max_lat-min_lat) + min_lat)
                    lon = float(arr_ds[idx,1]*(max_lon-min_lon) + min_lon)
                    sog = float(arr_ds[idx,2]*(max_sog-min_sog) + min_sog)
                    cog = float(arr_ds[idx,3]*(max_cog-min_cog) + min_cog)
                    rows.append((
                        str(voyage_id), idx, grid_ts[idx],
                        lat, lon, sog, cog, None,
                        gear_id[idx], changed[idx], t_since[idx]
                    ))

                cur.executemany("""
                    INSERT INTO ais_points
                        (voyage_id, seq_idx, ts, lat, lon, sog, cog, heading, gear_id, gear_changed_flag, time_since_change_min)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, rows)

                con.commit()
    print("Done inserting voyages, gear_events, ais_points.")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Match AIS & ERS and insert into Postgres schema")
    ap.add_argument("base_dir", type=str, help="Base directory (e.g., code/data)")
    ap.add_argument("year", type=str, help="Year to process (e.g., '2016')")
    ap.add_argument("--radio2mmsi", type=str, default="data/radio2mmsi.csv", help="callsign→MMSI map (CSV ';')")
    args = ap.parse_args()

    # Setup paths
    base_dir = os.path.abspath(args.base_dir)
    ais_dir = os.path.join(base_dir, args.year)
    ers_dir = os.path.join(base_dir, "ers", args.year)
    
    # Find AIS files
    ais_files = []
    if os.path.exists(ais_dir):
        for file in os.listdir(ais_dir):
            if file.endswith('.csv') and file != 'combined.csv':
                ais_files.append(os.path.join(ais_dir, file))
    
    # Find ERS file
    ers_file = None
    if os.path.exists(ers_dir):
        ers_files = [f for f in os.listdir(ers_dir) if f.endswith('.csv')]
        if ers_files:
            ers_file = os.path.join(ers_dir, ers_files[0])  # Take the first ERS file found

    # Validate found files
    if not ais_files:
        print(f"Error: No AIS files found in {ais_dir}")
        exit(1)
    if not ers_file:
        print(f"Error: No ERS file found in {ers_dir}")
        exit(1)

    # Sort AIS files to ensure consistent processing order
    ais_files.sort()

    print(f"\nProcessing data for year {args.year}")
    print(f"Found {len(ais_files)} AIS files:")
    for f in ais_files:
        print(f"  - {os.path.basename(f)}")
    print(f"ERS file:")
    print(f"  - {os.path.basename(ers_file)}")

    # Process the files
    try:
        match_and_insert(ais_files, ers_file, args.radio2mmsi)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        exit(1)