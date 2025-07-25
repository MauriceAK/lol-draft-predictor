import pandas as pd
import numpy as np
from typing import List
from collections import Counter

def process_lol_esports_data(file_paths: List[str], required_columns: List[str]) -> pd.DataFrame:
    """
    Processes raw League of Legends esports match data from multiple CSV files.
    ... (rest of the docstring is unchanged) ...
    """
    print("Starting data processing...")

    all_dfs = []
    for file_path in file_paths:
        try:
            # MODIFIED: Explicitly setting dtype for 'patch' to avoid mixed type warnings
            df = pd.read_csv(file_path, usecols=required_columns, dtype={'patch': str}, low_memory=False)
            all_dfs.append(df)
            print(f"Successfully loaded and added {file_path}.")
        except FileNotFoundError:
            print(f"Warning: The file {file_path} was not found. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}. Skipping.")
            continue

    if not all_dfs:
        print("Error: No dataframes were loaded. Exiting.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined all files. Total raw rows: {len(combined_df)}")

    combined_df = combined_df[combined_df['datacompleteness'] == 'complete'].copy()
    print(f"Filtered for complete data. Remaining rows: {len(combined_df)}")
    
    # Convert 'result' to numeric for aggregation
    combined_df['result'] = pd.to_numeric(combined_df['result'], errors='coerce')

    print("Aggregating player data into team data...")
    aggregated_df = combined_df.groupby(['gameid', 'side']).agg(
        league=('league', 'first'),
        year=('year', 'first'),
        patch=('patch', 'first'),
        team=('teamname', 'first'),
        players=('playername', list),
        champions=('champion', list),
        ban1=('ban1', 'first'),
        ban2=('ban2', 'first'),
        ban3=('ban3', 'first'),
        ban4=('ban4', 'first'),
        ban5=('ban5', 'first'),
        result=('result', 'first')
    ).reset_index()

    aggregated_df['bans'] = aggregated_df[['ban1', 'ban2', 'ban3', 'ban4', 'ban5']].values.tolist()
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
    processed_df.rename(columns={
        'blue_league': 'league',
        'blue_year': 'year',
        'blue_patch': 'patch'
    }, inplace=True)

    processed_df['winner'] = np.where(processed_df['blue_result'] == 1, 'Blue', 'Red')

    final_columns = [
        'gameid', 'league', 'year', 'patch', 'winner', 'blue_team', 'blue_players', 
        'blue_champions', 'blue_bans', 'red_team', 'red_players', 'red_champions', 'red_bans'
    ]
    processed_df = processed_df.reindex(columns=final_columns)

    print("Data processing complete.")
    return processed_df

def create_ml_features(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from processed data to create a model-ready dataset.
    """
    print("\nStarting feature engineering for machine learning...")
    
    df = processed_df.copy()
    
    # --- 1. Create Target Variable ---
    df['winner_is_blue'] = (df['winner'] == 'Blue').astype(int)

    # --- 2. Create Team Win Rate Features ---
    print("Calculating team win rates...")
    all_teams_blue = df[['blue_team', 'winner_is_blue']].rename(columns={'blue_team': 'team', 'winner_is_blue': 'won'})
    all_teams_red = df[['red_team', 'winner_is_blue']].rename(columns={'red_team': 'team', 'winner_is_blue': 'won'})
    all_teams_red['won'] = 1 - all_teams_red['won']
    
    team_games = pd.concat([all_teams_blue, all_teams_red])
    team_stats = team_games.groupby('team')['won'].agg(['sum', 'count']).rename(columns={'sum': 'wins', 'count': 'games'})
    team_stats['win_rate'] = team_stats['wins'] / team_stats['games']
    
    df = df.merge(team_stats[['win_rate']], left_on='blue_team', right_index=True).rename(columns={'win_rate': 'blue_team_win_rate'})
    df = df.merge(team_stats[['win_rate']], left_on='red_team', right_index=True).rename(columns={'win_rate': 'red_team_win_rate'})

    # --- 3. One-Hot Encode Champion Picks and Bans (Efficiently) ---
    print("One-hot encoding champion picks and bans...")

    for col in ['blue_champions', 'red_champions', 'blue_bans', 'red_bans']:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

    all_champions = set(c for col in ['blue_champions', 'red_champions', 'blue_bans', 'red_bans'] for c_list in df[col] for c in c_list if c is not None)
    
    # NEW (EFFICIENT) METHOD: Create all new columns as a list of Series first
    new_feature_columns = []
    
    for champion in all_champions:
        blue_pick_col = df['blue_champions'].apply(lambda picks: 1 if champion in picks else 0)
        blue_pick_col.name = f'blue_pick_{champion}'
        new_feature_columns.append(blue_pick_col)

        red_pick_col = df['red_champions'].apply(lambda picks: 1 if champion in picks else 0)
        red_pick_col.name = f'red_pick_{champion}'
        new_feature_columns.append(red_pick_col)
        
        blue_ban_col = df['blue_bans'].apply(lambda bans: 1 if champion in bans else 0)
        blue_ban_col.name = f'blue_ban_{champion}'
        new_feature_columns.append(blue_ban_col)

        red_ban_col = df['red_bans'].apply(lambda bans: 1 if champion in bans else 0)
        red_ban_col.name = f'red_ban_{champion}'
        new_feature_columns.append(red_ban_col)

    # Concatenate all new columns at once
    features_to_add = pd.concat(new_feature_columns, axis=1)
    df = pd.concat([df, features_to_add], axis=1)

    # --- 4. Finalize DataFrame ---
    print("Finalizing feature set...")
    columns_to_drop = [
        'gameid', 'league', 'year', 'patch', 'winner', 'blue_team', 'blue_players', 
        'blue_champions', 'blue_bans', 'red_team', 'red_players', 'red_champions', 'red_bans'
    ]
    features_df = df.drop(columns=columns_to_drop)
    
    print("Feature engineering complete.")
    return features_df


def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    """
    Saves a pandas DataFrame to a CSV file.
    ... (rest of the docstring is unchanged) ...
    """
    if dataframe.empty:
        print("Dataframe is empty. Nothing to save.")
        return

    try:
        dataframe.to_csv(output_path, index=False)
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")