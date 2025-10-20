-- PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- Reference: ports
CREATE TABLE IF NOT EXISTS ports (
  port_id       INTEGER PRIMARY KEY,
  name          TEXT NOT NULL,
  lat           DOUBLE PRECISION,
  lon           DOUBLE PRECISION,
  geom          geometry(Point, 4326)
);

-- Reference: vessels
CREATE TABLE IF NOT EXISTS vessels (
  mmsi          BIGINT PRIMARY KEY,
  callsign      TEXT,
  type          TEXT,
  flag          TEXT,
  length_m      DOUBLE PRECISION,
  width_m       DOUBLE PRECISION,
  draught_m     DOUBLE PRECISION,
  engine_kw     DOUBLE PRECISION,
  gross_tonnage DOUBLE PRECISION
);

-- Voyages (surrogate PK)
CREATE TABLE IF NOT EXISTS voyages (
  voyage_id     UUID PRIMARY KEY,
  mmsi          BIGINT NOT NULL REFERENCES vessels(mmsi),
  callsign      TEXT,
  start_ts      TIMESTAMPTZ NOT NULL,
  end_ts        TIMESTAMPTZ NOT NULL,
  dep_port_id   INTEGER REFERENCES ports(port_id),
  arr_port_id   INTEGER REFERENCES ports(port_id),
  label_port_id INTEGER REFERENCES ports(port_id),
  label_ts      TIMESTAMPTZ,
  gear_primary  TEXT,
  n_changes     INTEGER DEFAULT 0
);

-- AIS points (1:N with voyages)
CREATE TABLE IF NOT EXISTS ais_points (
  voyage_id     UUID NOT NULL REFERENCES voyages(voyage_id) ON DELETE CASCADE,
  seq_idx       INTEGER NOT NULL,
  ts            TIMESTAMPTZ NOT NULL,
  lat           DOUBLE PRECISION NOT NULL,
  lon           DOUBLE PRECISION NOT NULL,
  sog           REAL,
  cog           REAL,
  heading       REAL,
  gear_id       TEXT,
  gear_changed_flag BOOLEAN,
  time_since_change_min INTEGER,
  PRIMARY KEY (voyage_id, seq_idx),
  UNIQUE (voyage_id, ts)
);
CREATE INDEX IF NOT EXISTS ais_points_voy_ts_idx ON ais_points (voyage_id, ts);

-- Gear events (1:N with voyages)
CREATE TABLE IF NOT EXISTS gear_events (
  event_id      BIGSERIAL PRIMARY KEY,
  voyage_id     UUID NOT NULL REFERENCES voyages(voyage_id) ON DELETE CASCADE,
  ts            TIMESTAMPTZ NOT NULL,
  gear_code     TEXT NOT NULL,
  notes         TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS gear_events_voy_ts_uq ON gear_events (voyage_id, ts);
CREATE INDEX IF NOT EXISTS gear_events_voy_idx ON gear_events (voyage_id);

-- Enforce ais_points/gear_events timestamps inside voyage window
CREATE OR REPLACE FUNCTION enforce_ts_in_voyage()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM voyages v
    WHERE v.voyage_id = NEW.voyage_id
      AND NEW.ts >= v.start_ts
      AND NEW.ts <= v.end_ts
  ) THEN
    RAISE EXCEPTION 'Timestamp % not within voyage % window', NEW.ts, NEW.voyage_id;
  END IF;
  RETURN NEW;
END$$;

DROP TRIGGER IF EXISTS ais_points_ts_check ON ais_points;
CREATE TRIGGER ais_points_ts_check
BEFORE INSERT OR UPDATE ON ais_points
FOR EACH ROW EXECUTE FUNCTION enforce_ts_in_voyage();

DROP TRIGGER IF EXISTS gear_events_ts_check ON gear_events;
CREATE TRIGGER gear_events_ts_check
BEFORE INSERT OR UPDATE ON gear_events
FOR EACH ROW EXECUTE FUNCTION enforce_ts_in_voyage();