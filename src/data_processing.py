import pandas as pd
import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

def process_raw_data(file_paths: List[str], required_columns: List[str]) -> pd.DataFrame:
    """
    Processes raw, player-level match data into a game-per-row format.
    """
    print("--- Starting Initial Data Processing ---")
    all_dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, usecols=required_columns, dtype={'patch': str}, low_memory=False)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}. Error: {e}. Skipping.")
            continue

    if not all_dfs:
        print("Error: No dataframes were loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df[combined_df['datacompleteness'] == 'complete'].copy()
    combined_df['result'] = pd.to_numeric(combined_df['result'], errors='coerce')

    print("Aggregating player data into team data...")
    aggregated_df = combined_df.groupby(['gameid', 'side']).agg(
        league=('league', 'first'),
        year=('year', 'first'),
        patch=('patch', 'first'),
        team=('teamname', 'first'),
        players=('playername', list),
        champions=('champion', list),
        ban1=('ban1', 'first'), ban2=('ban2', 'first'), ban3=('ban3', 'first'),
        ban4=('ban4', 'first'), ban5=('ban5', 'first'),
        result=('result', 'first')
    ).reset_index()

    aggregated_df['bans'] = aggregated_df[['ban1', 'ban2', 'ban3', 'ban4', 'ban5']].apply(
        lambda x: [b for b in x if pd.notna(b)], axis=1
    )
    aggregated_df = aggregated_df.drop(columns=['ban1', 'ban2', 'ban3', 'ban4', 'ban5'])

    print("Pivoting data to have one row per game...")
    blue_team_data = aggregated_df[aggregated_df['side'] == 'Blue'].copy()
    red_team_data = aggregated_df[aggregated_df['side'] == 'Red'].copy()

    blue_team_data.rename(columns=lambda col: f"blue_{col}" if col not in ['gameid', 'side'] else col, inplace=True)
    red_team_data.rename(columns=lambda col: f"red_{col}" if col not in ['gameid', 'side'] else col, inplace=True)
    
    processed_df = pd.merge(
        blue_team_data.drop(columns=['side']),
        red_team_data.drop(columns=['side', 'red_league', 'red_year', 'red_patch']),
        on='gameid'
    )
    processed_df.rename(columns={'blue_league': 'league', 'blue_year': 'year', 'blue_patch': 'patch'}, inplace=True)
    processed_df['winner'] = np.where(processed_df['blue_result'] == 1, 'Blue', 'Red')

    final_columns = [
        'gameid', 'league', 'year', 'patch', 'winner', 'blue_team', 'blue_players', 
        'blue_champions', 'blue_bans', 'red_team', 'red_players', 'red_champions', 'red_bans'
    ]
    
    for col in ['blue_players', 'blue_champions', 'blue_bans', 'red_players', 'red_champions', 'red_bans']:
        processed_df[col] = processed_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        processed_df[col] = processed_df[col].apply(lambda lst: [item for item in lst if pd.notna(item)])

    print("Initial data processing complete.")
    return processed_df.reindex(columns=final_columns)


def _precompute_stats(df: pd.DataFrame) -> Dict:
    """
    Pre-computes historical statistics on a per-patch basis to be used for point-in-time feature creation.
    """
    print("Pre-computing historical statistics per patch...")
    
    stats = {
        'team_wins_per_patch': defaultdict(lambda: defaultdict(int)),
        'team_games_per_patch': defaultdict(lambda: defaultdict(int)),
        'player_champ_wins_per_patch': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'player_champ_games_per_patch': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'player_team_champ_wins_per_patch': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'player_team_champ_games_per_patch': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'champ_patch_wins': defaultdict(lambda: defaultdict(int)),
        'champ_patch_games': defaultdict(lambda: defaultdict(int)),
    }

    for _, row in df.iterrows():
        patch, blue_won = row['patch'], row['winner'] == 'Blue'
        
        stats['team_wins_per_patch'][row['blue_team']][patch] += 1 if blue_won else 0
        stats['team_games_per_patch'][row['blue_team']][patch] += 1
        stats['team_wins_per_patch'][row['red_team']][patch] += 0 if blue_won else 1
        stats['team_games_per_patch'][row['red_team']][patch] += 1
        
        for i in range(len(row['blue_players'])):
            p, c, t = row['blue_players'][i], row['blue_champions'][i], row['blue_team']
            if p and c:
                stats['player_champ_wins_per_patch'][p][c][patch] += 1 if blue_won else 0
                stats['player_champ_games_per_patch'][p][c][patch] += 1
                stats['player_team_champ_wins_per_patch'][(p, t)][c][patch] += 1 if blue_won else 0
                stats['player_team_champ_games_per_patch'][(p, t)][c][patch] += 1
        
        for i in range(len(row['red_players'])):
            p, c, t = row['red_players'][i], row['red_champions'][i], row['red_team']
            if p and c:
                stats['player_champ_wins_per_patch'][p][c][patch] += 0 if blue_won else 1
                stats['player_champ_games_per_patch'][p][c][patch] += 1
                stats['player_team_champ_wins_per_patch'][(p, t)][c][patch] += 0 if blue_won else 1
                stats['player_team_champ_games_per_patch'][(p, t)][c][patch] += 1

        for c in row['blue_champions']:
            if c:
                stats['champ_patch_wins'][c][patch] += 1 if blue_won else 0
                stats['champ_patch_games'][c][patch] += 1
        for c in row['red_champions']:
            if c:
                stats['champ_patch_wins'][c][patch] += 0 if blue_won else 1
                stats['champ_patch_games'][c][patch] += 1

    print("Finished pre-computing stats.")
    return stats

def create_ml_features(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from processed data, including advanced point-in-time stats to prevent data leakage.
    """
    print("\nStarting advanced feature engineering...")
    
    df = processed_df.copy().sort_values(by=['year', 'patch']).reset_index(drop=True)
    patches = sorted(df['patch'].unique())
    
    historical_stats = _precompute_stats(df)
    new_features_list = []

    print("Generating point-in-time features for each game...")
    for index, row in df.iterrows():
        current_patch_index = patches.index(row['patch'])
        past_patches = patches[:current_patch_index]
        
        game_features = {}
        
        # Team winrate over last 3 patches (form)
        form_patches = patches[max(0, current_patch_index - 3) : current_patch_index]
        for side in ['blue', 'red']:
            team = row[f'{side}_team']
            wins = sum(historical_stats['team_wins_per_patch'][team].get(p, 0) for p in form_patches)
            games = sum(historical_stats['team_games_per_patch'][team].get(p, 0) for p in form_patches)
            game_features[f'{side}_team_winrate_form'] = wins / games if games > 5 else 0.5

        # Player and Champion specific winrates (using all past data)
        for side in ['blue', 'red']:
            for i in range(len(row[f'{side}_players'])):
                p, c, t = row[f'{side}_players'][i], row[f'{side}_champions'][i], row[f'{side}_team']
                
                p_c_wins = sum(historical_stats['player_champ_wins_per_patch'][p][c].get(p_patch, 0) for p_patch in past_patches)
                p_c_games = sum(historical_stats['player_champ_games_per_patch'][p][c].get(p_patch, 0) for p_patch in past_patches)
                game_features[f'{side}_p{i+1}_champ_wr'] = p_c_wins / p_c_games if p_c_games > 3 else 0.5
                
                p_t_c_wins = sum(historical_stats['player_team_champ_wins_per_patch'][(p, t)][c].get(p_patch, 0) for p_patch in past_patches)
                p_t_c_games = sum(historical_stats['player_team_champ_games_per_patch'][(p, t)][c].get(p_patch, 0) for p_patch in past_patches)
                game_features[f'{side}_p{i+1}_team_champ_wr'] = p_t_c_wins / p_t_c_games if p_t_c_games > 2 else 0.5
        
        # --- DATA LEAK FIX ---
        # Champion winrate on current patch, excluding the current game's result
        for side in ['blue', 'red']:
            for i, c in enumerate(row[f'{side}_champions']):
                # Get total stats for the champion on this patch
                total_wins = historical_stats['champ_patch_wins'][c].get(row['patch'], 0)
                total_games = historical_stats['champ_patch_games'][c].get(row['patch'], 0)
                
                # Determine if the champion in *this* game won
                this_game_won = (side == 'blue' and row['winner'] == 'Blue') or \
                                (side == 'red' and row['winner'] == 'Red')
                
                # Subtract this game's outcome from the totals
                wins_excluding_this_game = total_wins - 1 if this_game_won else total_wins
                games_excluding_this_game = total_games - 1
                
                # Calculate win rate based on other games
                game_features[f'{side}_c{i+1}_patch_wr'] = wins_excluding_this_game / games_excluding_this_game if games_excluding_this_game > 0 else 0.5

        new_features_list.append(game_features)

    new_features_df = pd.DataFrame(new_features_list, index=df.index)

    print("One-hot encoding base features...")
    df['winner_is_blue'] = (df['winner'] == 'Blue').astype(int)
    target_variable = df[['winner_is_blue']]
    
    blue_team_dummies = pd.get_dummies(df['blue_team'], prefix='blue_team', dtype=int)
    red_team_dummies = pd.get_dummies(df['red_team'], prefix='red_team', dtype=int)
    league_dummies = pd.get_dummies(df['league'], prefix='league', dtype=int)
    
    mlb = MultiLabelBinarizer()
    blue_pick_features = pd.DataFrame(mlb.fit_transform(df['blue_champions']), columns=[f'blue_pick_{cls}' for cls in mlb.classes_], index=df.index)
    red_pick_features = pd.DataFrame(mlb.fit_transform(df['red_champions']), columns=[f'red_pick_{cls}' for cls in mlb.classes_], index=df.index)
    blue_ban_features = pd.DataFrame(mlb.fit_transform(df['blue_bans']), columns=[f'blue_ban_{cls}' for cls in mlb.classes_], index=df.index)
    red_ban_features = pd.DataFrame(mlb.fit_transform(df['red_bans']), columns=[f'red_ban_{cls}' for cls in mlb.classes_], index=df.index)

    print("Finalizing feature set...")
    final_df = pd.concat([
        target_variable, new_features_df, league_dummies, blue_team_dummies, 
        red_team_dummies, blue_pick_features, red_pick_features,
        blue_ban_features, red_ban_features
    ], axis=1)
    
    # Drop columns with all zero values, which can happen with sparse one-hot encoding
    final_df = final_df.loc[:, (final_df != 0).any(axis=0)]
    
    print("Advanced feature engineering complete.")
    return final_df
