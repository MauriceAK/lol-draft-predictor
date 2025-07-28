import duckdb
import json
import pandas as pd
import config

# --- SCRIPT SETUP ---
# Set Pandas to display all columns so we don't miss any data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Data Inspection Utility ---")

# --- LOAD NECESSARY FILES ---
try:
    with open('scripts/team_mapping.json', 'r') as f:
        team_mapping = json.load(f)
    with open('scripts/draft_live.json', 'r') as f:
        draft = json.load(f)
except FileNotFoundError as e:
    print(f"ERROR: Could not find a necessary file: {e.filename}")
    exit()

# --- CONNECT TO DATABASE ---
try:
    con = duckdb.connect(config.DB_PATH, read_only=True)
    print(f"Successfully connected to database at: {config.DB_PATH}\n")
except Exception as e:
    print(f"ERROR: Could not connect to the database. Make sure the path in config.py is correct. Details: {e}")
    exit()

# --- INSPECT DATA FOR EACH TEAM IN DRAFT ---
for team_color in ['blue_team', 'red_team']:
    team_name = draft[team_color]['name']
    team_id = team_mapping.get(team_name)

    if not team_id:
        print(f"WARNING: Team '{team_name}' not found in team_mapping.json. Skipping.")
        continue

    print(f"--- Inspecting Data for: {team_name} (ID: {team_id}) ---")

    # 1. Inspect the TEAM_PERFORMANCE table
    print("\n[1] Checking 'team_performance' table...")
    try:
        # This is where 'team_kills15' comes from
        perf_query = "SELECT * FROM team_performance WHERE teamid = ? ORDER BY game_date DESC"
        perf_df = con.execute(perf_query, [team_id]).df()

        if perf_df.empty:
            print(f"RESULT: NO DATA FOUND for this team in 'team_performance'. This explains why 'team_kills15' would be NaN.")
        else:
            print("RESULT: Found the following performance data (most recent first):")
            print(perf_df.head(10)) # Show the 10 most recent games

    except duckdb.CatalogException:
        print("ERROR: The table 'team_performance' does not exist in your database.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    # 2. Inspect the TEAM_FORM table
    print("\n[2] Checking 'team_form' table...")
    try:
        # This is where '_last10' stats come from
        form_query = "SELECT * FROM team_form WHERE teamid = ? ORDER BY game_date DESC"
        form_df = con.execute(form_query, [team_id]).df()

        if form_df.empty:
            print(f"RESULT: NO DATA FOUND for this team in 'team_form'. This explains why all '_last10' stats are default values (0.5 or 0.0).")
        else:
            print("RESULT: Found the following form data (most recent first):")
            print(form_df.head(10))

    except duckdb.CatalogException:
        print("ERROR: The table 'team_form' does not exist in your database.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n" + "="*50 + "\n")

con.close()