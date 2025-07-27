#!/usr/bin/env bash
set -euo pipefail

DB="${1:-data/lol.duckdb}"

echo "▶ Populating DuckDB at $DB …"

duckdb "$DB" <<'SQL'
-- 1) Load raw CSVs
CREATE OR REPLACE TABLE raw_matches AS
  SELECT *
  FROM read_csv_auto('data/2024_LoL_esports_match_data_from_OraclesElixir.csv', HEADER, AUTO_DETECT);

-- If you have a 2025 incremental file, repeat for that:
-- INSERT INTO raw_matches
--   SELECT * FROM read_csv_auto('data/2025_LoL_esports_match_data_from_OraclesElixir.csv', HEADER, AUTO_DETECT);

-- 2) Build team_matches
.read scripts/sql/schema_raw.sql

-- 3) Build champion_meta
.read scripts/sql/create_champion_meta.sql

-- 4) Build champion_synergy
.read scripts/sql/create_champion_synergy.sql

-- 5) Build champion_counters
.read scripts/sql/create_champion_counters.sql

-- 6) Build player_champ_stats
.read scripts/sql/create_player_champ_stats.sql

-- 7) Build team_performance (if separate)
.read scripts/sql/create_team_performance.sql

-- 8) Build team_form
.read scripts/sql/create_team_form.sql
SQL

echo "✅ Done."
