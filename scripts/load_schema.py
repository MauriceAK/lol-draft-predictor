#!/usr/bin/env python3
# scripts/load_schema.py

import duckdb
import glob
import os
import config

def main():
    # Connect to DuckDB
    db_path = config.DB_PATH
    conn = duckdb.connect(db_path)
    print(f"🔄 Loading schema into {db_path}…")

    # Find all SQL files under scripts/sql in the desired order
    sql_dir = os.path.join("scripts", "sql")
    patterns = [
        "schema_raw.sql",
        "create_team_stats.sql",
        "create_match_stats.sql",
        "create_champion_meta.sql",
        "create_champion_synergy.sql",
        "create_champion_counters.sql",
        "create_player_champion_stats.sql",
        "create_team_performance.sql",
        "create_team_form.sql"
    ]

    for fn in patterns:
        path = os.path.join(sql_dir, fn)
        if not os.path.exists(path):
            print(f"❗ Warning: {path} not found, skipping")
            continue
        print(f"→ executing {fn}")
        sql = open(path, "r").read()
        conn.execute(sql)

    print(f"✅ Schema loaded successfully into {db_path}")

if __name__ == "__main__":
    main()
