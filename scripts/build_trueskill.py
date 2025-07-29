import duckdb
import pandas as pd
from trueskill import Rating, rate, setup
import config
import sys
from datetime import datetime
import numpy as np

# --- SETUP ---
setup(mu=1500, sigma=500, beta=400, tau=5, draw_probability=0)

# --- MODEL CONSTANTS ---
INTERNATIONAL_LEAGUES = ['MSI', 'WLDs', 'EWC']
MINIMUM_GAMES_PLAYED = 10
TIME_DECAY_RATE = 0.0019
SIGMA_INFLATION_FACTOR = 100.0
# The amount to increase sigma by for a seasonal reset.
SEASONAL_RESET_SIGMA_INCREASE = 150.0 

def get_db_connection(read_only=False):
    """Establishes connection to the DuckDB database."""
    return duckdb.connect(config.DB_PATH, read_only=read_only)

def build_trueskill_table():
    """
    Calculates historical TrueSkill ratings using a time-decay, sigma-inflation,
    and seasonal-reset model.
    """
    con = get_db_connection(read_only=True)

    games_played_query = "SELECT teamid, COUNT(DISTINCT gameid) as games FROM raw_player_stats GROUP BY teamid HAVING games >= ?;"
    reliable_teams_df = con.execute(games_played_query, [MINIMUM_GAMES_PLAYED]).df()
    reliable_teams = set(reliable_teams_df['teamid'])
    print(f"Identified {len(reliable_teams)} reliable teams.", file=sys.stderr)

    games_df = con.execute("SELECT * FROM game_results_for_trueskill;").df()
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    con.close()
    
    latest_date = games_df['game_date'].max()

    ratings = {}
    history = []
    last_game_year = {}

    print("Calculating TrueSkill ratings with Time-Decay, Sigma-Inflation, and Seasonal Resets...", file=sys.stderr)
    for _, row in games_df.iterrows():
        winner, loser = row['winner_id'], row['loser_id']
        current_game_year = row['game_date'].year

        if winner in reliable_teams and loser in reliable_teams:
            # --- SEASONAL RESET LOGIC ---
            for team_id in [winner, loser]:
                if team_id in last_game_year and last_game_year[team_id] < current_game_year:
                    # New season for this team, apply a soft reset by increasing sigma
                    old_rating = ratings.get(team_id, Rating())
                    ratings[team_id] = Rating(mu=old_rating.mu, sigma=old_rating.sigma + SEASONAL_RESET_SIGMA_INCREASE)
                last_game_year[team_id] = current_game_year

            winner_rating = ratings.get(winner, Rating())
            loser_rating = ratings.get(loser, Rating())

            history.append((row['match_id'], winner, row['game_date'], winner_rating.mu, winner_rating.sigma))
            history.append((row['match_id'], loser, row['game_date'], loser_rating.mu, loser_rating.sigma))

            # --- HYBRID MODEL IMPLEMENTATION ---
            days_ago = (latest_date - row['game_date']).days
            time_weight = np.exp(-TIME_DECAY_RATE * days_ago)
            is_international = row['league'] in INTERNATIONAL_LEAGUES
            
            final_winner_rating, final_loser_rating = winner_rating, loser_rating
            if is_international:
                final_winner_rating = Rating(mu=winner_rating.mu, sigma=winner_rating.sigma + SIGMA_INFLATION_FACTOR)
                final_loser_rating = Rating(mu=loser_rating.mu, sigma=loser_rating.sigma + SIGMA_INFLATION_FACTOR)
            
            base_weight = 2.0 if is_international else 1.0
            game_weight = base_weight * time_weight

            (new_winner_rating,), (new_loser_rating,) = rate(
                rating_groups=[(final_winner_rating,), (final_loser_rating,)],
                weights=[(game_weight,), (game_weight,)]
            )
            ratings[winner], ratings[loser] = new_winner_rating, new_loser_rating

    history_df = pd.DataFrame(history, columns=['match_id', 'teamid', 'game_date', 'trueskill_mu', 'trueskill_sigma'])

    con = get_db_connection(read_only=False)
    print(f"Writing {len(history_df)} rating records to the database...", file=sys.stderr)
    con.execute("DROP TABLE IF EXISTS team_trueskill_ratings;")
    con.execute("CREATE TABLE team_trueskill_ratings AS SELECT * FROM history_df;")
    con.close()

    print("Successfully created 'team_trueskill_ratings' table with final model.", file=sys.stderr)

if __name__ == "__main__":
    build_trueskill_table()