# lolpredictor/etl.py

import duckdb
import pandas as pd
import config
from trueskill import Rating

def get_connection():
    return duckdb.connect(config.DB_PATH, read_only=True)


def load_and_prep_data():
    """High-level function to perform all data loading and preparation."""
    con = get_connection()


    # Load other necessary tables, removing the old form/elo tables
    tables = {
        "raw_matches": con.execute("SELECT * FROM match_picks_with_opponents").df(),
        "meta": con.execute("SELECT * FROM champion_meta").df(),
        "synergy": con.execute("SELECT * FROM champion_synergy").df(),
        "counters": con.execute("SELECT * FROM champion_counters").df(),
        "player_stats": con.execute("SELECT * FROM player_champion_stats").df(),
        "performance": con.execute("SELECT * FROM team_performance").df(),
        "trueskill": con.execute("SELECT * FROM team_trueskill_ratings").df()
    }
    con.close()

        # Convert game_date to datetime objects for correct sorting
    tables['trueskill']['game_date'] = pd.to_datetime(tables['trueskill']['game_date'])
    tables['performance']['game_date'] = pd.to_datetime(tables['performance']['game_date'])


    maps = build_feature_maps(tables)
    match_df = create_match_pairs(tables, maps)
    return match_df, maps


def build_feature_maps(tables):
    """Builds lookup dictionaries for efficient feature retrieval."""
    maps = {
        "win_map": {(r.patch, r.champion): r.win_rate for r in tables["meta"].itertuples()},
        "syn_map": {tuple(sorted((r.champ1, r.champ2))): r.synergy_win_rate for r in tables["synergy"].itertuples()},
        "ctr_map": {(r.champ_a, r.champ_b): r.counter_win_rate for r in tables["counters"].itertuples()},
        "pc_map": {(r.player_id, r.champion): r for r in tables["player_stats"].itertuples()},
        "perf_map": {(r.match_id, r.teamid): r for r in tables["performance"].itertuples()},
        # Create a map for historical TrueSkill lookups
        "trueskill_map": {(r.match_id, r.teamid): r for r in tables["trueskill"].itertuples()}
    }

    # Create a map of the absolute most recent stats for each team for live predictions
    if not tables["trueskill"].empty:
        # FIX: Sort by 'game_date' instead of 'match_id' for true chronological order.
        latest_trueskill_df = tables["trueskill"].sort_values('game_date').drop_duplicates(subset='teamid', keep='last')
        maps["latest_trueskill_map"] = latest_trueskill_df.set_index('teamid').to_dict('index')

    if not tables["performance"].empty:
        # FIX: Ensure this also sorts by 'game_date' for consistency.
        latest_perf_df = tables["performance"].sort_values('game_date').drop_duplicates(subset='teamid', keep='last')
        maps["latest_perf_map"] = latest_perf_df.set_index('teamid').to_dict('index')

    maps['raw_matches'] = tables['raw_matches']

    return maps

def build_team_vector(team_id, match_id, patch, team_picks, opp_picks, team_players, maps):
    """Assembles a feature vector for a single team in a match."""
    feat = {}
    default_rating = Rating() # Provides mu=1500, sigma=500

    # Draft and player features
    for i, champ in enumerate(team_picks, 1):
        feat[f'pick{i}_win_rate'] = maps['win_map'].get((patch, champ), 0.5)
    syn_vals = [maps['syn_map'].get(tuple(sorted((c1, c2))), 0.5) for i, c1 in enumerate(team_picks) for j, c2 in enumerate(team_picks) if i < j and c1 and c2]
    ctr_vals = [maps['ctr_map'].get((p, o), 0.5) for p in team_picks for o in opp_picks if p and o]
    feat['team_synergy_mean'] = sum(syn_vals) / len(syn_vals) if syn_vals else 0.5
    feat['counter_mean'] = sum(ctr_vals) / len(ctr_vals) if ctr_vals else 0.5
    for i, (pid, champ) in enumerate(zip(team_players, team_picks), 1):
        stats_dict = maps['pc_map'].get((pid, champ), {})
        if hasattr(stats_dict, "_asdict"): stats_dict = stats_dict._asdict()
        feat[f'pick{i}_player_wr'] = stats_dict.get('player_champ_win_rate', 0.5)
        feat[f'pick{i}_player_games'] = stats_dict.get('player_champ_games', 0)

    # --- SIMPLIFIED FEATURE LOOKUP ---
    if match_id != 0: # Historical game
        trueskill_stats = maps['trueskill_map'].get((match_id, team_id), {})
        perf = maps['perf_map'].get((match_id, team_id), {})
    else: # Live game
        trueskill_stats = maps.get('latest_trueskill_map', {}).get(team_id, {})
        perf = maps.get('latest_perf_map', {}).get(team_id, {})

    if hasattr(trueskill_stats, "_asdict"): trueskill_stats = trueskill_stats._asdict()
    if hasattr(perf, "_asdict"): perf = perf._asdict()

    feat['trueskill_mu'] = trueskill_stats.get('trueskill_mu', default_rating.mu)
    feat['trueskill_sigma'] = trueskill_stats.get('trueskill_sigma', default_rating.sigma)
    #feat['team_kills15'] = perf.get('team_kills15', 0.0) # Keep playstyle indicator

    return feat

def create_match_pairs(tables, maps):
    raw_df = tables["raw_matches"]
    processed_matches = []
    # Process each match_id only once to avoid duplicating pairs
    for match_id in raw_df['match_id'].unique():
        match_group = raw_df[raw_df['match_id'] == match_id]
        if len(match_group) != 2: continue

        team1_row, team2_row = match_group.iloc[0], match_group.iloc[1]
        patch = team1_row['patch']

        t1_id, t2_id = team1_row['teamid'], team2_row['teamid']
        t1_picks = [team1_row[f'pick{i}'] for i in range(1, 6)]
        t1_players = [team1_row[f'player{i}_id'] for i in range(1, 6)]
        t2_picks = [team2_row[f'pick{i}'] for i in range(1, 6)]
        t2_players = [team2_row[f'player{i}_id'] for i in range(1, 6)]

        v1 = build_team_vector(t1_id, match_id, patch, t1_picks, t2_picks, t1_players, maps)
        v2 = build_team_vector(t2_id, match_id, patch, t2_picks, t1_picks, t2_players, maps)

        processed_matches.append({
            "match_id": match_id,
            "team1_vector": v1,
            "team2_vector": v2,
            "label": int(team1_row['label'])
        })
    return pd.DataFrame(processed_matches)


if __name__ == "__main__":
    # You can still use this for standalone debugging if you wish
    print("Running ETL in standalone mode...")
    match_df, maps = load_and_prep_data()
    print("\n--- ETL Process Complete ---")
    print(f"Successfully processed {len(match_df)} matches.")
    if not match_df.empty:
        print("Feature vector for the first match:")
        print(match_df.iloc[0]['team1_vector'])
    print("\n'latest_trueskill_map' sample:")
    print(list(maps['latest_trueskill_map'].items())[:5])
    # if hasattr(perf, "_asdict"):
    #     perf = perf._asdict()
    # if hasattr(form, "_asdict"):
    #     form = form._asdict()