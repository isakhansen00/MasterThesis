import sqlite3

import sqlite3

# 1) Create / open the DB file
con = sqlite3.connect("ais.db")
cur = con.cursor()

# 2) Enforce foreign keys for this connection
cur.execute("PRAGMA foreign_keys = ON;")

# 3) Execute your schema (paste the full CREATE TABLE/INDEX/VIEW SQL here)
schema_sql = """
PRAGMA foreign_keys = ON;

-- === PORT ===
CREATE TABLE IF NOT EXISTS port (
  port_id        INTEGER PRIMARY KEY,
  name           TEXT NOT NULL UNIQUE,
  country        TEXT,
  latitude_deg   REAL NOT NULL,
  longitude_deg  REAL NOT NULL,
  timezone       TEXT,
  created_at_utc TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- === STATUS ===
CREATE TABLE IF NOT EXISTS status (
  status_id   INTEGER PRIMARY KEY,
  code        TEXT NOT NULL UNIQUE,
  description TEXT
);

-- === VESSEL ===
CREATE TABLE IF NOT EXISTS vessel (
  vessel_id      INTEGER PRIMARY KEY,
  mmsi           INTEGER UNIQUE,
  imo            INTEGER UNIQUE,
  name           TEXT,
  vessel_type    TEXT,
  length_m       REAL,
  tonnage_gt     REAL,
  engine_kw      REAL,
  flag           TEXT,
  year_built     INTEGER,
  home_port_id   INTEGER,
  created_at_utc TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  FOREIGN KEY (home_port_id) REFERENCES port(port_id)
);

-- === VESSEL_MOVEMENT ===
CREATE TABLE IF NOT EXISTS vessel_movement (
  movement_id       INTEGER PRIMARY KEY,
  vessel_id         INTEGER NOT NULL,
  ts_utc            TEXT    NOT NULL,
  latitude_deg      REAL    NOT NULL,
  longitude_deg     REAL    NOT NULL,
  sog_kn            REAL,
  cog_deg           REAL,
  heading_deg       REAL,
  nav_status        TEXT,
  status_id         INTEGER,
  declared_dest     TEXT,
  eta_utc           TEXT,
  arrival_port_id   INTEGER,
  arrival_voyage_id INTEGER,
  created_at_utc    TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  FOREIGN KEY (vessel_id)       REFERENCES vessel(vessel_id),
  FOREIGN KEY (status_id)       REFERENCES status(status_id),
  FOREIGN KEY (arrival_port_id) REFERENCES port(port_id)
);

-- === VOYAGE (optional) ===
CREATE TABLE IF NOT EXISTS voyage (
  voyage_id           INTEGER PRIMARY KEY,
  vessel_id           INTEGER NOT NULL,
  start_ts_utc        TEXT NOT NULL,
  end_ts_utc          TEXT,
  origin_port_id      INTEGER,
  destination_port_id INTEGER,
  FOREIGN KEY (vessel_id)           REFERENCES vessel(vessel_id),
  FOREIGN KEY (origin_port_id)      REFERENCES port(port_id),
  FOREIGN KEY (destination_port_id) REFERENCES port(port_id)
);

-- Helpful indices
CREATE INDEX IF NOT EXISTS idx_port_name           ON port(name);
CREATE INDEX IF NOT EXISTS idx_status_code         ON status(code);
CREATE INDEX IF NOT EXISTS idx_vessel_mmsi         ON vessel(mmsi);
CREATE INDEX IF NOT EXISTS idx_vessel_type         ON vessel(vessel_type);
CREATE INDEX IF NOT EXISTS idx_move_vessel_ts      ON vessel_movement(vessel_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_move_arrival_port   ON vessel_movement(arrival_port_id);
CREATE INDEX IF NOT EXISTS idx_move_status         ON vessel_movement(status_id);
CREATE INDEX IF NOT EXISTS idx_move_latlon         ON vessel_movement(latitude_deg, longitude_deg);
CREATE INDEX IF NOT EXISTS idx_voyage_vessel       ON voyage(vessel_id);
CREATE INDEX IF NOT EXISTS idx_voyage_dest         ON voyage(destination_port_id);
"""
cur.executescript(schema_sql)

# 4) Seed minimal lookup values
cur.executescript("""
INSERT OR IGNORE INTO status (code, description) VALUES
  ('FISHING','Actively engaged in fishing operations'),
  ('TRANSITING','Underway, not fishing'),
  ('AT_ANCHOR','Anchored or moored'),
  ('IN_PORT','Alongside within a port area');

INSERT OR IGNORE INTO port (name, country, latitude_deg, longitude_deg, timezone) VALUES
  ('Bergen','Norway',60.39299,5.32415,'Europe/Oslo'),
  ('Stavanger','Norway',58.97005,5.73332,'Europe/Oslo'),
  ('Tromsø','Norway',69.64920,18.95532,'Europe/Oslo');
""")

con.commit()
con.close()