import duckdb
import json
import pandas as pd
import config
import os

def get_team_rankings():
    """
    Connects to the database, retrieves the latest TrueSkill ratings,
    and combines them with team names for a readable report.
    """
    team_mapping_path = 'scripts/team_mapping.json'
    db_path = config.DB_PATH

    # 1. Load and invert the team mapping for easy lookup
    if not os.path.exists(team_mapping_path):
        print(f"Error: Team mapping file not found at {team_mapping_path}")
        return
        
    with open(team_mapping_path, 'r') as f:
        team_mapping = json.load(f)
    # [cite_start]The original mapping is {teamname: teamid}[cite: 36]. We invert it.
    id_to_name_map = {v: k for k, v in team_mapping.items()}

    # 2. Define the SQL query to get the latest rating for each team
    query = """
    WITH MajorLeagueTeams AS (
        -- First, create a list of all team IDs that have played in the specified major leagues
        SELECT DISTINCT teamid
        FROM raw_player_stats
        WHERE league IN ('LCS', 'LCK', 'LPL', 'LEC', 'LTA S', 'LTA N', 'LCP', 'CBLOL')
    ),
    RankedRatings AS (
        -- Then, number the ratings for ONLY those teams
        SELECT
            teamid,
            trueskill_mu,
            ROW_NUMBER() OVER(PARTITION BY teamid ORDER BY game_date DESC) as rn
        FROM
            team_trueskill_ratings
        WHERE
            teamid IN (SELECT teamid FROM MajorLeagueTeams)
    )
    -- Finally, select the most recent rating for each team and join to get the name
    SELECT
        teamid,
        trueskill_mu
    FROM
        RankedRatings
    WHERE
        rn = 1
    ORDER BY
        trueskill_mu DESC;
    """

    # 3. Connect to the database and execute the query
    try:
        con = duckdb.connect(db_path, read_only=True)
        results_df = con.execute(query).df()
        con.close()
    except Exception as e:
        print(f"An error occurred while querying the database: {e}")
        return

    # 4. Map team IDs to names and display the results
    results_df['Team Name'] = results_df['teamid'].map(id_to_name_map).fillna('Unknown Team')
    
    # Format the DataFrame for better presentation
    final_rankings = results_df[['Team Name', 'trueskill_mu']]
    final_rankings.rename(columns={'trueskill_mu': 'TrueSkill Rating'}, inplace=True)
    final_rankings.index += 1 # Start ranking from 1 instead of 0
    
    print("--- Global Team TrueSkill Rankings ---")
    print(final_rankings.to_string())


if __name__ == '__main__':
    get_team_rankings()