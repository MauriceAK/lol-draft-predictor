from collections import defaultdict
import pandas as pd
from trueskill import Rating, rate_1vs1, setup

def update_trueskill(winner_rating, loser_rating, is_international=False):
    default_beta = 4.1667
    beta = default_beta * 2 if is_international else default_beta
    setup(beta=beta)
    return rate_1vs1(winner_rating, loser_rating)

def track_elo_per_game(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['year', 'patch']).reset_index(drop=True)
    team_ratings = defaultdict(lambda: Rating())
    regular_leagues = {'LCK', 'LPL', 'LEC', 'LCS'}

    match_elos = []

    for _, row in df.iterrows():
        blue, red = row['blue_team'], row['red_team']
        winner = row['winner']
        league = row['league']
        is_intl = league not in regular_leagues

        blue_rating = team_ratings[blue]
        red_rating = team_ratings[red]

        match_elos.append({
            'gameid': row['gameid'],
            'blue_mu': blue_rating.mu,
            'blue_sigma': blue_rating.sigma,
            'red_mu': red_rating.mu,
            'red_sigma': red_rating.sigma,
            'elo_gap': blue_rating.mu - red_rating.mu,
            'elo_conf_gap': (blue_rating.mu / blue_rating.sigma) - (red_rating.mu / red_rating.sigma),
            'win_prob_mu_only': 1 / (1 + 10 ** ((red_rating.mu - blue_rating.mu) / 400))
        })

        if winner == 'Blue':
            new_blue, new_red = update_trueskill(blue_rating, red_rating, is_intl)
        else:
            new_red, new_blue = update_trueskill(red_rating, blue_rating, is_intl)

        team_ratings[blue] = new_blue
        team_ratings[red] = new_red

    return pd.DataFrame(match_elos)
