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
Improved AIS-ERS preprocessing that properly handles multiple ERS file types.
Based on Andreas Løvland's code with enhancements for multi-file ERS processing.
"""


geod = Geod(ellps='WGS84')

# ---------------- normalization bounds (region-specific) ----------------
min_cog, max_cog = 0.0, 360.0
min_sog, max_sog = 0, 30.0
min_lat, max_lat = 69.2, 73.0
min_lon, max_lon = 13.0, 31.5

# ---------------- column aliases for multi-file ERS ----------------
ALIASES = {
    "callsign": ["Radiokallesignal (ERS)", "Radiokallesignal", "Callsign", "call_sign", "CALLSIGN"],
    "type":     ["Meldingstype", "Type", "MsgType"],
    "stop_ts":  ["Stopptidspunkt", "Tidspunkt", "Timestamp", "Tid"],
    "arrive_ts":["Ankomsttidspunkt", "Ankomst", "ETA"],
    "depart_ts":["Avgangstidspunkt", "Avgang", "ETD"],
    "gear":     ["Redskap FAO", "Redskap FDIR", "Redskap", "Gear"],
    # static (appears in all ERS types)
    "length_m":      ["Største lengde", "Fartøylengde", "Lengde", "length_m"],
    "width_m":       ["Bredde", "width_m"],
    "draught_m":     ["Dypgående", "Dypgaende", "draught_m"],
    "engine_kw":     ["Motorkraft", "engine_kw"],
    "gross_tonnage": ["Bruttotonnasje 1969", "Bruttotonnasje annen", "GT", "gross_tonnage"],
    "vessel_type":   ["Fartøytype", "VesselType", "vessel_type"],
    "flag":          ["Fartøynasjonalitet (kode)", "Flag", "flag"],
    "vessel_id":     ["Fartøy ID"],
    "port_code":     ["Havn (kode)"],
    "port_name":     ["Havn"]
}

GEAR_REPORT_DELAY_MIN = 60  # conservative to avoid leakage

def pick_col(df: pd.DataFrame, names):
    """Find first matching column name from aliases."""
    for n in names:
        if n in df.columns: return n
    return None

def haversine_km(lon1, lat1, lon2, lat2):
    """Calculate distance in km using Haversine formula."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def interpolate(t: int, track: np.ndarray):
    """Interpolate AIS position at timestamp t using geodesic interpolation."""
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
    """Downsample AIS track to fixed time intervals."""
    TS = 4
    sampling_track = np.empty((0, 6))
    for t in range(int(arr[0, TS]), int(arr[-1, TS]), minutes*60):
        interpolated = interpolate(t, arr)
        if interpolated is None: return None
        sampling_track = np.vstack([sampling_track, interpolated])
    return sampling_track

def filter_outlier_messages(arr: np.ndarray) -> np.ndarray:
    """Remove tracks with large temporal gaps (>2 hours)."""
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
    """Return most common element in sequence."""
    seq = [s for s in seq if pd.notnull(s)]
    return Counter(seq).most_common(1)[0][0] if seq else None

def build_gear_timeline(dca_df, callsign, t_start, t_end, delay_min=GEAR_REPORT_DELAY_MIN):
    """Build timeline of gear changes from DCA reports."""
    gear_col = pick_col(dca_df, ALIASES["gear"])
    call_col = pick_col(dca_df, ALIASES["callsign"])
    time_col = pick_col(dca_df, ALIASES["stop_ts"])
    
    if not all([gear_col, call_col, time_col]): return []
    
    df = dca_df.loc[
        (dca_df[call_col] == callsign) &
        (dca_df[time_col] >= t_start) &
        (dca_df[time_col] <= t_end) &
        (dca_df[gear_col].notna())
    ].copy()
    
    if df.empty: return []
    
    # Add delay to avoid data leakage
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
    """Stamp gear information onto interpolated AIS points."""
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
    """Connect to PostgreSQL using environment variables."""
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
    """Insert or update vessel static information."""
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
    """Load ports from database."""
    if not PSYCOPG_AVAILABLE:
        raise ImportError("psycopg is not installed. Install with: pip install psycopg[binary]")
    with con.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT port_id, name, lat, lon FROM ports")
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["port_id","name","lat","lon"])

def load_ports_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load ports from CSV file.
    Handles both formats:
    - Standard: port_id,name,lat,lon
    - Custom: Port_ID,Code,Name,City,Nationality,Latitude,Longitude
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ports CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Normalize column names to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Map different column name variations to standard names
    column_mapping = {}
    
    # Port ID mapping
    if 'port_id' not in df.columns:
        for col in ['portid', 'id', 'port_number']:
            if col in df.columns:
                column_mapping[col] = 'port_id'
                break
    
    # Name mapping
    if 'name' not in df.columns:
        for col in ['port_name', 'portname']:
            if col in df.columns:
                column_mapping[col] = 'name'
                break
    
    # Latitude mapping
    if 'lat' not in df.columns:
        for col in ['latitude', 'lat_dd', 'y']:
            if col in df.columns:
                column_mapping[col] = 'lat'
                break
    
    # Longitude mapping
    if 'lon' not in df.columns:
        for col in ['longitude', 'long', 'lon_dd', 'x']:
            if col in df.columns:
                column_mapping[col] = 'lon'
                break
    
    # Apply mappings
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Validate required columns
    required = ['port_id', 'lat', 'lon']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Ports CSV missing required columns: {missing}. Found: {df.columns.tolist()}")
    
    # Filter out ports with missing coordinates
    before_count = len(df)
    df = df.dropna(subset=['lat', 'lon'])
    df = df[(df['lat'] != 0) | (df['lon'] != 0)]  # Remove (0,0) coordinates unless it's a valid port
    after_count = len(df)
    
    if before_count > after_count:
        print(f"[!] Filtered out {before_count - after_count} ports with missing/invalid coordinates")
    
    # Ensure port_id is integer
    df['port_id'] = df['port_id'].astype(int)
    
    return df[['port_id', 'name', 'lat', 'lon']] if 'name' in df.columns else df[['port_id', 'lat', 'lon']]

def nearest_port_id(lat, lon, ports_df):
    """Find nearest port to given coordinates."""
    if ports_df.empty: return None, None
    dists = haversine_km(lon, lat, ports_df["lon"].values, ports_df["lat"].values)
    idx = int(np.argmin(dists))
    return int(ports_df.iloc[idx]["port_id"]), float(dists[idx])

# ---------------- ERS processing ----------------
def load_ers_files(ers_dir, year):
    """
    Load and process all ERS files (DCA, POR, DEP) for a given year.
    Returns separate DataFrames for each type.
    Skips TRA (overforingsmelding) files as they're not needed.
    """
    dca_df = None
    por_df = None
    dep_df = None
    
    for filename in os.listdir(ers_dir):
        if not filename.endswith('.csv'):
            continue
        
        # Skip TRA (transfer) messages - not needed
        if 'overforingsmelding' in filename.lower() or 'tra' in filename.lower():
            print(f"[*] Skipping TRA file: {filename}")
            continue
            
        filepath = os.path.join(ers_dir, filename)
        
        # Identify file type by name
        if 'fangstmelding' in filename.lower() or 'dca' in filename.lower():
            print(f"[*] Loading DCA file: {filename}")
            dca_df = pd.read_csv(filepath, delimiter=";", low_memory=False)
        elif 'ankomstmelding' in filename.lower() or 'por' in filename.lower():
            print(f"[*] Loading POR file: {filename}")
            por_df = pd.read_csv(filepath, delimiter=";", low_memory=False)
        elif 'avgangsmelding' in filename.lower() or 'dep' in filename.lower():
            print(f"[*] Loading DEP file: {filename}")
            dep_df = pd.read_csv(filepath, delimiter=";", low_memory=False)
    
    # Parse timestamps with explicit Norwegian date format (DD.MM.YYYY HH:MM:SS)
    if dca_df is not None:
        stop_col = pick_col(dca_df, ALIASES["stop_ts"])
        if stop_col:
            dca_df[stop_col] = pd.to_datetime(dca_df[stop_col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    
    if por_df is not None:
        arrive_col = pick_col(por_df, ALIASES["arrive_ts"])
        if arrive_col:
            por_df[arrive_col] = pd.to_datetime(por_df[arrive_col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    
    if dep_df is not None:
        depart_col = pick_col(dep_df, ALIASES["depart_ts"])
        if depart_col:
            dep_df[depart_col] = pd.to_datetime(dep_df[depart_col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    
    return dca_df, por_df, dep_df

def get_static_from_ers(ers_dfs, callsign):
    """
    Extract static vessel information from any available ERS file.
    ers_dfs: tuple of (dca_df, por_df, dep_df)
    """
    static_info = {
        "length_m": None, "width_m": None, "draught_m": None,
        "engine_kw": None, "gross_tonnage": None, "vessel_type": None, "flag": None
    }
    
    # Try each dataframe in order
    for df in ers_dfs:
        if df is None:
            continue
            
        call_col = pick_col(df, ALIASES["callsign"])
        if call_col is None:
            continue
            
        vessel_df = df.loc[df[call_col] == callsign].copy()
        if vessel_df.empty:
            continue
        
        # Extract each static field
        for key in static_info.keys():
            if static_info[key] is not None:  # Already found
                continue
                
            col = pick_col(vessel_df, ALIASES[key])
            if col is None:
                continue
                
            # Get most recent non-null value
            for v in vessel_df[col]:
                if pd.notnull(v):
                    try:
                        if key in ["vessel_type", "flag"]:
                            static_info[key] = str(v)
                        else:
                            static_info[key] = float(v)
                        break
                    except:
                        continue
    
    return static_info

def match_voyage_from_por_dep(por_df, dep_df, callsign, dca_stop_time, time_window_hours=168):
    """
    Match DCA fishing event to corresponding POR/DEP events.
    Returns (arrival_time, departure_time, arrival_port_id, departure_port_id) or None.
    """
    call_col_por = pick_col(por_df, ALIASES["callsign"])
    call_col_dep = pick_col(dep_df, ALIASES["callsign"])
    arrive_col = pick_col(por_df, ALIASES["arrive_ts"])
    depart_col = pick_col(dep_df, ALIASES["depart_ts"])
    
    if not all([call_col_por, call_col_dep, arrive_col, depart_col]):
        return None
    
    # Find arrivals after DCA stop (within time window)
    arrivals = por_df.loc[
        (por_df[call_col_por] == callsign) &
        (por_df[arrive_col] > dca_stop_time) &
        (por_df[arrive_col] <= dca_stop_time + pd.Timedelta(hours=time_window_hours))
    ].sort_values(arrive_col)
    
    if arrivals.empty:
        return None
    
    arrival_time = arrivals.iloc[0][arrive_col]
    
    # Find corresponding departure (closest after arrival)
    departures = dep_df.loc[
        (dep_df[call_col_dep] == callsign) &
        (dep_df[depart_col] >= arrival_time)
    ].sort_values(depart_col)
    
    if departures.empty:
        return None
    
    departure_time = departures.iloc[0][depart_col]
    
    return arrival_time, departure_time

# ---------------- main ----------------
def load_and_concat_ais_files(ais_files):
    """Load and concatenate multiple AIS files into a single DataFrame."""
    dfs = []
    for file in ais_files:
        # Skip merged.csv to avoid duplicate data
        if 'merged' in os.path.basename(file).lower():
            print(f"[*] Skipping merged file: {os.path.basename(file)}")
            continue
        print(f"[*] Loading AIS file: {os.path.basename(file)}")
        df = pd.read_csv(file, delimiter=",", low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def index_ais_by_mmsi(ais_df):
    """
    Create an indexed dictionary for fast MMSI lookups.
    Returns: dict[mmsi] -> DataFrame of AIS records for that MMSI
    """
    print("[*] Indexing AIS data by MMSI for fast lookups...")
    ais_grouped = {}
    for mmsi, group in ais_df.groupby('mmsi'):
        # Pre-sort by date for faster filtering
        ais_grouped[mmsi] = group.sort_values('date').copy()
    print(f"[+] Indexed {len(ais_grouped)} unique vessels")
    return ais_grouped

def match_and_insert(ais_files: list, ers_dir: str, year: str, radio2mmsi_path="data/radio2mmsi.csv", 
                     output_mode="db", csv_output_dir="output", ports_csv_path="data/ports.csv", limit=None):
    """
    Main processing function: match AIS tracks to ERS events and save to DB or CSV.
    
    Args:
        ais_files: List of AIS CSV file paths
        ers_dir: Directory containing ERS files
        year: Year being processed
        radio2mmsi_path: Path to callsign-to-MMSI mapping file
        output_mode: "db" for database, "csv" for CSV files
        csv_output_dir: Directory to save CSV files (only used if output_mode="csv")
        ports_csv_path: Path to ports CSV file (required for CSV mode, optional for DB mode)
        limit: Optional limit on number of DCA records to process (for testing)
    """
    print("[*] Loading datasets")
    ais_df = load_and_concat_ais_files(ais_files)
    dca_df, por_df, dep_df = load_ers_files(ers_dir, year)
    radio2mmsi = pd.read_csv(radio2mmsi_path, skiprows=1, delimiter=";", index_col=0).squeeze().to_dict()
    print("[+] Datasets loaded")
    
    if dca_df is None:
        raise ValueError("No DCA file found - required for voyage processing")
    
    # Parse AIS timestamps and filter by year range for performance
    print("[*] Parsing AIS timestamps and filtering by year...")
    ais_df["date"] = pd.to_datetime(ais_df["date"], errors="coerce")
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{int(year)+1}-01-01")
    ais_df = ais_df[(ais_df["date"] >= year_start) & (ais_df["date"] < year_end)].copy()
    print(f"[+] Filtered to {len(ais_df):,} AIS records in year {year}")
    
    # OPTIMIZATION: Index AIS by MMSI for O(1) lookups instead of O(n) filtering
    ais_by_mmsi = index_ais_by_mmsi(ais_df)
    
    # Get column names (cache these lookups)
    call_col = pick_col(dca_df, ALIASES["callsign"])
    stop_col = pick_col(dca_df, ALIASES["stop_ts"])
    
    if call_col is None or stop_col is None:
        raise ValueError("DCA file missing required columns")
    
    # Apply limit if specified (for testing)
    if limit is not None:
        print(f"[!] LIMIT: Processing only first {limit} DCA records (test mode)")
        dca_df = dca_df.head(limit)
    
    static_cache = {}
    
    # Setup for CSV output mode
    if output_mode == "csv":
        os.makedirs(csv_output_dir, exist_ok=True)
        vessels_list = []
        voyages_list = []
        gear_events_list = []
        ais_points_list = []
        # Load ports from CSV file
        ports_df = load_ports_from_csv(ports_csv_path)
        print(f"[*] CSV output mode - will save to {csv_output_dir}/")
        print(f"[*] Loaded {len(ports_df)} ports from {ports_csv_path}")
    else:
        # Database mode - connect and load ports
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "Database mode requires psycopg. Install with:\n"
                "  pip install psycopg[binary]\n"
                "Or use CSV mode: --output csv"
            )
        con = connect_env()
        ports_df = load_ports_df(con)
        print("[*] Database mode - connected to PostgreSQL")
        print(f"[*] Loaded {len(ports_df)} ports from database")
    
    try:
        # Process each DCA (fishing operation) event
        for _, dca_row in tqdm(dca_df.iterrows(), total=len(dca_df.index), desc="Processing voyages"):
            callsign = dca_row[call_col]
            stop_time = dca_row[stop_col]
            
            if pd.isna(stop_time) or pd.isna(callsign):
                continue
            
            mmsi = radio2mmsi.get(callsign, 0)
            if not mmsi:
                continue
            
            # Try to match with POR/DEP if available
            arrival_time = None
            departure_time = None
            
            if por_df is not None and dep_df is not None:
                match = match_voyage_from_por_dep(por_df, dep_df, callsign, stop_time)
                if match:
                    arrival_time, departure_time = match
            
            # Define voyage time range
            # If we have POR/DEP: from last DCA to midpoint between arrival and departure
            # If not: use a default window (e.g., 7 days before DCA stop)
            if arrival_time and departure_time:
                start_time = stop_time - pd.Timedelta(days=7)  # Look back for AIS
                end_time = arrival_time + (departure_time - arrival_time) / 2
            else:
                # Fallback: use DCA stop time with reasonable window
                start_time = stop_time - pd.Timedelta(days=7)
                end_time = stop_time + pd.Timedelta(days=1)
            
            # OPTIMIZATION: Extract AIS track using pre-indexed data (much faster!)
            if mmsi not in ais_by_mmsi:
                continue
            
            df = ais_by_mmsi[mmsi]
            df = df.loc[
                (df["date"] > start_time) &
                (df["date"] < end_time)
            ].copy()
            
            if df.empty:
                continue
            
            df["timestamp"] = df["date"].astype(np.int64) // 10**9
            df = df.loc[:, ["lat", "long", "sog", "cog", "timestamp", "mmsi"]]
            arr = df.to_numpy()
            
            # Filter outliers
            arr = filter_outlier_messages(arr)
            if len(arr) == 0:
                continue
            
            # Quality filters
            if len(arr) < 20 or (datetime.fromtimestamp(arr[-1][-2]) - datetime.fromtimestamp(arr[0][-2]) < timedelta(hours=4)):
                continue
            
            # Downsample to 5-minute intervals
            arr_ds = downsample(arr, minutes=5)
            if arr_ds is None or len(arr_ds) == 0:
                continue
            
            # Normalize coordinates
            arr_ds[:,0] = (arr_ds[:,0].astype(np.float32) - min_lat)/(max_lat - min_lat)
            arr_ds[:,1] = (arr_ds[:,1].astype(np.float32) - min_lon)/(max_lon - min_lon)
            arr_ds[:,2] = (arr_ds[:,2].astype(np.float32) - min_sog)/(max_sog - min_sog)
            arr_ds[:,3] = (arr_ds[:,3].astype(np.float32) - min_cog)/(max_cog - min_cog)
            arr_ds[:, :4][arr_ds[:, :4] >= 1] = 0.9999
            
            # Build gear timeline from DCA
            timeline = build_gear_timeline(dca_df, callsign, start_time, end_time)
            gear_primary = most_common([g for _,g in timeline]) if timeline else None
            n_changes = max(0, len(timeline)-1)
            
            # Label: last position and nearest port
            last_norm = arr_ds[-1]
            lat_real = float(last_norm[0]*(max_lat-min_lat) + min_lat)
            lon_real = float(last_norm[1]*(max_lon-min_lon) + min_lon)
            
            # Find nearest port (works for both DB and CSV modes now)
            if not ports_df.empty:
                label_port_id, port_distance = nearest_port_id(lat_real, lon_real, ports_df)
            else:
                label_port_id = None
                port_distance = None
            
            label_ts = datetime.fromtimestamp(int(arr_ds[-1][4]))
            
            # Get static vessel info
            if callsign not in static_cache:
                static_cache[callsign] = get_static_from_ers((dca_df, por_df, dep_df), callsign)
            static_info = static_cache[callsign]
            
            # Generate voyage ID
            voyage_id = str(uuid.uuid4())
            
            if output_mode == "db":
                # Insert to database
                with con.cursor() as cur:
                    upsert_vessel(cur, mmsi, callsign, static_info)
                    
                    cur.execute("""
                        INSERT INTO voyages (voyage_id, mmsi, callsign, start_ts, end_ts,
                                            dep_port_id, arr_port_id, label_port_id, label_ts,
                                            gear_primary, n_changes)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        voyage_id, int(mmsi), callsign,
                        start_time.to_pydatetime(), end_time.to_pydatetime(),
                        None, None, label_port_id, label_ts, gear_primary, n_changes
                    ))
                    
                    # Gear events
                    if timeline:
                        cur.executemany(
                            "INSERT INTO gear_events (voyage_id, ts, gear_code) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                            [(voyage_id, t, g) for (t, g) in timeline],
                        )
                    
                    # AIS points
                    grid_ts = [datetime.fromtimestamp(int(ts)) for ts in arr_ds[:,4]]
                    gear_id, changed, t_since = stamp_gear_on_grid(grid_ts, timeline)
                    
                    rows = []
                    for idx in range(len(grid_ts)):
                        lat = float(arr_ds[idx,0]*(max_lat-min_lat) + min_lat)
                        lon = float(arr_ds[idx,1]*(max_lon-min_lon) + min_lon)
                        sog = float(arr_ds[idx,2]*(max_sog-min_sog) + min_sog)
                        cog = float(arr_ds[idx,3]*(max_cog-min_cog) + min_cog)
                        rows.append((
                            voyage_id, idx, grid_ts[idx],
                            lat, lon, sog, cog, None,
                            gear_id[idx], changed[idx], t_since[idx]
                        ))
                    
                    cur.executemany("""
                        INSERT INTO ais_points
                            (voyage_id, seq_idx, ts, lat, lon, sog, cog, heading, gear_id, gear_changed_flag, time_since_change_min)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, rows)
                    
                    con.commit()
            
            else:  # CSV mode
                # Collect vessel info
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
                # Check if vessel already in list
                if not any(v["mmsi"] == int(mmsi) for v in vessels_list):
                    vessels_list.append(vessel_row)
                
                # Collect voyage info
                voyage_row = {
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
                    "n_ais_points": len(arr_ds)
                }
                voyages_list.append(voyage_row)
                
                # Collect gear events
                for t, g in timeline:
                    gear_events_list.append({
                        "voyage_id": voyage_id,
                        "ts": t,
                        "gear_code": g
                    })
                
                # Collect AIS points
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
                        "lat": lat,
                        "lon": lon,
                        "sog": sog,
                        "cog": cog,
                        "heading": None,
                        "gear_id": gear_id[idx],
                        "gear_changed_flag": changed[idx],
                        "time_since_change_min": t_since[idx]
                    })
        
        # Finalize output
        if output_mode == "db":
            con.close()
            print("Done inserting voyages, gear_events, ais_points to database.")
        else:
            # Save to CSV files
            print(f"\n[*] Saving {len(vessels_list)} vessels to CSV...")
            pd.DataFrame(vessels_list).to_csv(
                os.path.join(csv_output_dir, f"vessels_{year}.csv"), 
                index=False
            )
            
            print(f"[*] Saving {len(voyages_list)} voyages to CSV...")
            pd.DataFrame(voyages_list).to_csv(
                os.path.join(csv_output_dir, f"voyages_{year}.csv"), 
                index=False
            )
            
            print(f"[*] Saving {len(gear_events_list)} gear events to CSV...")
            if gear_events_list:
                pd.DataFrame(gear_events_list).to_csv(
                    os.path.join(csv_output_dir, f"gear_events_{year}.csv"), 
                    index=False
                )
            
            print(f"[*] Saving {len(ais_points_list)} AIS points to CSV...")
            pd.DataFrame(ais_points_list).to_csv(
                os.path.join(csv_output_dir, f"ais_points_{year}.csv"), 
                index=False
            )
            
            # Save port mapping for model training reference
            print(f"[*] Saving port mapping to CSV...")
            ports_df.to_csv(
                os.path.join(csv_output_dir, f"ports_{year}.csv"),
                index=False
            )
            
            print(f"[+] Done! CSV files saved to {csv_output_dir}/")
            print(f"    - {len(ports_df)} ports (for model num_classes)")
            print(f"    - {len(vessels_list)} unique vessels")
            print(f"    - {len(voyages_list)} voyages with labeled destinations")
    
    finally:
        # Clean up database connection if it was opened
        if output_mode == "db" and 'con' in locals():
            try:
                con.close()
            except:
                pass

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Match AIS & ERS and insert into Postgres schema or save to CSV")
    ap.add_argument("base_dir", type=str, help="Base directory (e.g., src/data)")
    ap.add_argument("year", type=str, help="Year to process (e.g., '2016')")
    ap.add_argument("--radio2mmsi", type=str, default=None, help="callsign→MMSI map (default: BASE_DIR/radio2mmsi.csv)")
    ap.add_argument("--output", type=str, choices=["db", "csv"], default="db", 
                    help="Output mode: 'db' for PostgreSQL database, 'csv' for CSV files")
    ap.add_argument("--csv-dir", type=str, default="output", 
                    help="Directory to save CSV files (only used with --output csv)")
    ap.add_argument("--ports-csv", type=str, default=None,
                    help="Path to ports CSV file (default: BASE_DIR/ports.csv, required for CSV mode)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit processing to first N DCA records (useful for testing, e.g., --limit 100)")
    args = ap.parse_args()
    
    # Set defaults relative to base_dir if not provided
    if args.radio2mmsi is None:
        args.radio2mmsi = os.path.join(args.base_dir, "radio2mmsi.csv")
    if args.ports_csv is None:
        args.ports_csv = os.path.join(args.base_dir, "ports.csv")
    
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
    
    if not ais_files:
        print(f"Error: No AIS files found in {ais_dir}")
        exit(1)
    
    if not os.path.exists(ers_dir):
        print(f"Error: ERS directory not found: {ers_dir}")
        exit(1)
    
    # Validate ports file for CSV mode
    if args.output == "csv" and not os.path.exists(args.ports_csv):
        print(f"Error: Ports CSV file not found: {args.ports_csv}")
        print("This file is required for CSV mode to label destination ports.")
        exit(1)
    
    ais_files.sort()
    
    print(f"\nProcessing data for year {args.year}")
    print(f"Output mode: {args.output.upper()}")
    if args.output == "csv":
        print(f"CSV output directory: {os.path.abspath(args.csv_dir)}")
        print(f"Ports reference file: {args.ports_csv}")
    if args.limit:
        print(f"LIMIT: Processing only first {args.limit} DCA records")
    print(f"Found {len(ais_files)} AIS files:")
    for f in ais_files:
        print(f"  - {os.path.basename(f)}")
    
    try:
        match_and_insert(ais_files, ers_dir, args.year, args.radio2mmsi, 
                        output_mode=args.output, csv_output_dir=args.csv_dir,
                        ports_csv_path=args.ports_csv, limit=args.limit)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
