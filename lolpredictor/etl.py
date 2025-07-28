# lolpredictor/etl.py

import duckdb
import pandas as pd
import config

def get_connection():
    return duckdb.connect(config.DB_PATH, read_only=True)

def load_all_tables(con):
    # This reads from the corrected, descriptive view name
    raw_query = "SELECT * FROM match_picks_with_opponents"
    tables = {
        "raw_matches": con.execute(raw_query).df(),
        "meta": con.execute("SELECT * FROM champion_meta").df(),
        "synergy": con.execute("SELECT * FROM champion_synergy").df(),
        "counters": con.execute("SELECT * FROM champion_counters").df(),
        "player_stats": con.execute("SELECT * FROM player_champion_stats").df(),
        "performance": con.execute("SELECT * FROM team_performance").df(),
        "form": con.execute("SELECT * FROM team_form").df(),
    }
    return tables

def build_feature_maps(tables):
    """
    Builds lookup dictionaries (maps) for efficient feature retrieval.
    The key change is creating a simple, robust 'latest_form_map' for live predictions.
    """
    # Create the base maps from tables
    maps = {
        "win_map": {(r.patch, r.champion): r.win_rate for r in tables["meta"].itertuples()},
        "syn_map": {tuple(sorted((r.champ1, r.champ2))): r.synergy_win_rate for r in tables["synergy"].itertuples()},
        "ctr_map": {(r.champ_a, r.champ_b): r.counter_win_rate for r in tables["counters"].itertuples()},
        "pc_map": {(r.player_id, r.champion): r for r in tables["player_stats"].itertuples()},
        "perf_map": {(r.match_id, r.teamid): r for r in tables["performance"].itertuples()},
        "form_map": {(r.match_id, r.teamid): r for r in tables["form"].itertuples()}
    }

    # --- ROBUST LATEST STATS LOGIC ---
    # Create a map of the absolute most recent form stats for each team.
    # This is used for live predictions when we don't have a historical match_id.
    if not tables["form"].empty:
        # Sort by date, drop duplicates, keeping only the last entry for each team
        print(tables["form"].head())  # Debugging line to check form data
        latest_form_df = tables["form"].sort_values('game_date').drop_duplicates(subset='teamid', keep='last')
        maps["latest_form_map"] = latest_form_df.set_index('teamid').to_dict('index')

    if not tables["performance"].empty:
        latest_perf_df = tables["performance"].sort_values('game_date').drop_duplicates(subset='teamid', keep='last')
        maps["latest_perf_map"] = latest_perf_df.set_index('teamid').to_dict('index')
    # ------------------------------------

    return maps

def build_team_vector(team_id, match_id, patch, team_picks, opp_picks, team_players, maps):
    feat = {}

    for i, champ in enumerate(team_picks, 1):
        feat[f'pick{i}_win_rate'] = maps['win_map'].get((patch, champ), 0.5)

    syn_vals = [maps['syn_map'].get(tuple(sorted((c1, c2))), 0.5) for i, c1 in enumerate(team_picks) for j, c2 in enumerate(team_picks) if i < j and c1 and c2]
    ctr_vals = [maps['ctr_map'].get((p, o), 0.5) for p in team_picks for o in opp_picks if p and o]
    feat['team_synergy_mean'] = sum(syn_vals) / len(syn_vals) if syn_vals else 0.5
    feat['counter_mean'] = sum(ctr_vals) / len(ctr_vals) if ctr_vals else 0.5

    for i, (pid, champ) in enumerate(zip(team_players, team_picks), 1):
        stats = maps['pc_map'].get((pid, champ))
        stats_dict = stats._asdict() if hasattr(stats, '_asdict') else (stats or {})
        feat[f'pick{i}_player_wr'] = stats_dict.get('player_champ_win_rate', 0.5)
        feat[f'pick{i}_player_games'] = stats_dict.get('player_champ_games', 0)

    # --- ROBUST FEATURE LOOKUP ---
    # For historical games (training/eval), use the exact match data.
    # For live games (match_id=0), use the new 'latest' maps.
    if match_id != 0:
        perf = maps['perf_map'].get((match_id, team_id), {})
        form = maps['form_map'].get((match_id, team_id), {})
    else:
        perf = maps.get('latest_perf_map', {}).get(team_id, {})
        form = maps.get('latest_form_map', {}).get(team_id, {})

    if hasattr(perf, "_asdict"):
        perf = perf._asdict()
    if hasattr(form, "_asdict"):
        form = form._asdict()
        
    # Ensure we use defaults if any lookup fails
    feat['win_rate_last10'] = form.get('win_rate_last10', 0.5)
    feat['avg_golddiff15_last10'] = form.get('avg_golddiff15_last10', 0.0)
    feat['team_kills15'] = perf.get('team_kills15', 0.0)
    return feat

def create_match_pairs(tables, maps):
    raw_df = tables["raw_matches"]
    processed_matches = []
    for match_id, group in raw_df.groupby("match_id"):
        if len(group) != 2: continue
        
        team1_row, team2_row = group.iloc[0], group.iloc[1]
        
        t1_picks = [team1_row[f'pick{i}'] for i in range(1, 6)]
        t1_players = [team1_row[f'player{i}_id'] for i in range(1, 6)]
        t2_picks = [team2_row[f'pick{i}'] for i in range(1, 6)]
        t2_players = [team2_row[f'player{i}_id'] for i in range(1, 6)]

        v1 = build_team_vector(team1_row.teamid, match_id, team1_row.patch, t1_picks, t2_picks, t1_players, maps)
        v2 = build_team_vector(team2_row.teamid, match_id, team2_row.patch, t2_picks, t1_picks, t2_players, maps)
        
        processed_matches.append({
            "match_id": match_id, "team1_vector": v1, "team2_vector": v2, "label": int(team1_row.label)
        })
    return pd.DataFrame(processed_matches)

def load_and_prep_data():
    """High-level function to perform all data loading and preparation."""
    con = get_connection()
    tables = load_all_tables(con)
    con.close()
    maps = build_feature_maps(tables)
    match_df = create_match_pairs(tables, maps)
    return match_df, maps

def __main__():
    """Main function for standalone execution."""
    match_df, maps = load_and_prep_data()
    print(maps['pc_map'])
    print(f"Loaded {len(match_df)} matches with {len(maps)} feature maps.")
    return match_df, maps

if __name__ == "__main__":  
    __main__()

    # if hasattr(perf, "_asdict"):
    #     perf = perf._asdict()
    # if hasattr(form, "_asdict"):
    #     form = form._asdict()