#!/usr/bin/env python3
# scripts/generate_team_mapping.py

import duckdb
import json
import os

def main():
    # 1) Connect to your DuckDB
    con = duckdb.connect("data/lol.duckdb")

    # 2) Pull distinct teamname → teamid (as strings)
    df = con.execute("""
        SELECT DISTINCT teamname, teamid
        FROM raw_player_stats
        WHERE teamname IS NOT NULL
    """).df()

    # 3) Build mapping dict (no int() conversion)
    mapping = {row.teamname: row.teamid for row in df.itertuples()}

    # 4) Write to JSON
    os.makedirs("scripts", exist_ok=True)
    out_path = "scripts/team_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"✅ Wrote {len(mapping)} entries to {out_path}")

if __name__ == "__main__":
    main()
