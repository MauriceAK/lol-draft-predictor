import pandas as pd
import numpy as np
import json
from typing import List, Dict
from collections import defaultdict

def process_raw_data(file_paths: List[str], required_columns: List[str]) -> pd.DataFrame:
    all_dfs = [pd.read_csv(file, low_memory=False) for file in file_paths]
    df = pd.concat(all_dfs, ignore_index=True)
    print("Unique values in 'datacompleteness':", df['datacompleteness'].unique())
    df['datacompleteness'] = df['datacompleteness'].astype(str).str.strip().str.lower()
    df = df[df['datacompleteness'] == 'complete']
    print("Rows after filtering:", len(df))

    side_counts = df.groupby('gameid')['side'].nunique()
    valid_gameids = side_counts[side_counts == 2].index
    df = df[df['gameid'].isin(valid_gameids)]

    # Keep patch info from any row per gameid (player rows have it)
    patch_per_game = df.groupby('gameid')['patch'].first()

    team_rows = df[df['position'] == 'team']
    blue_side = team_rows[team_rows['side'] == 'Blue'].set_index('gameid')
    red_side = team_rows[team_rows['side'] == 'Red'].set_index('gameid')

    pivoted_df = blue_side.join(red_side, lsuffix='_blue', rsuffix='_red')
    print("Shape after join:", pivoted_df.shape)
    if pivoted_df.empty:
        print("Pivoted DataFrame is empty after join. Exiting.")
        return pd.DataFrame()

    pivoted_df['winner_is_blue'] = (pivoted_df['result_blue'] == 1).astype(int)
    # Preserve metadata columns from blue_side (like league, patch, gamedate)
    for col in ['patch', 'league', 'gamedate']:
        if col in blue_side.columns:
            pivoted_df[col] = blue_side[col]
    # Add patch info back
    pivoted_df['patch'] = patch_per_game

    for side in ['Blue', 'Red']:
        side_lower = side.lower()
        players = df[(df['side'] == side) & (df['position'] != 'team')].groupby('gameid')['playername'].apply(list)
        champions = df[(df['side'] == side) & (df['position'] != 'team')].groupby('gameid')['champion'].apply(list)
        bans = df[(df['side'] == side) & (df['position'] == 'team')].set_index('gameid')[['ban1','ban2','ban3','ban4','ban5']]

        pivoted_df[f'{side_lower}_players'] = players
        pivoted_df[f'{side_lower}_champions'] = champions
        pivoted_df[f'{side_lower}_bans'] = bans.apply(lambda x: x.tolist(), axis=1)

    return pivoted_df.reset_index()


def _precompute_stats(df: pd.DataFrame) -> Dict:
    print("Pre-computing historical statistics per patch...")
    stats = {
        'champ_patch_winrate': defaultdict(lambda: defaultdict(float)),
    }

    champ_wins = defaultdict(lambda: defaultdict(int))
    champ_games = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        patch = row['patch']
        blue_won = row['winner_is_blue'] == 1
        for c in row['blue_champions']:
            champ_games[c][patch] += 1
            if blue_won:
                champ_wins[c][patch] += 1
        for c in row['red_champions']:
            champ_games[c][patch] += 1
            if not blue_won:
                champ_wins[c][patch] += 1

    for c in champ_games:
        for patch in champ_games[c]:
            total = champ_games[c][patch]
            wins = champ_wins[c][patch]
            stats['champ_patch_winrate'][c][patch] = wins / total if total > 0 else 0.5

    print("Finished pre-computing stats.")
    return stats

def create_feature_mappings(df: pd.DataFrame, output_path: str):
    all_champions = set(c for col in ['blue_champions', 'red_champions', 'blue_bans', 'red_bans'] for sublist in df[col].dropna() for c in sublist)
    all_teams = set(df['teamname_blue'].unique()) | set(df['teamname_red'].unique())
    all_champions = [c for c in all_champions if isinstance(c, str)]

    champ_map = {name: i + 1 for i, name in enumerate(sorted(all_champions))}
    team_map = {name: i + 1 for i, name in enumerate(sorted(list(all_teams)))}

    mappings = {'champion_map': champ_map, 'team_map': team_map}
    with open(output_path, 'w') as f:
        json.dump(mappings, f)
    return mappings

def transform_data_for_nn(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    df_transformed = df.copy()
    champ_map = mappings['champion_map']
    team_map = mappings['team_map']

    df_transformed['blue_team_idx'] = df_transformed['teamname_blue'].map(team_map).fillna(0).astype(int)
    df_transformed['red_team_idx'] = df_transformed['teamname_red'].map(team_map).fillna(0).astype(int)

    for col in ['blue_champions', 'red_champions']:
        df_transformed[f'{col}_idx'] = df_transformed[col].apply(
            lambda champs: [champ_map.get(c, 0) for c in champs] if isinstance(champs, list) else [0]*5
        )
    return df_transformed
