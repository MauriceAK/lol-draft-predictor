# notebooks/data_exploration.py
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
# ---

# %% [markdown]
# # LoL Draft Predictor — Data Exploration
# This notebook rebuilds the DuckDB schema, peeks at each feature table,
# and runs quick train+eval for all iterations so you can compare side-by-side.

# %% [markdown]
# ## 1) Rebuild all SQL tables

# %%
import subprocess

# Point this at your duckdb file
DB="data/lol.duckdb"

for f in [
    "scripts/sql/schema_raw.sql",
    "scripts/sql/create_team_stats.sql",
    "scripts/sql/create_match_stats.sql",
    "scripts/sql/create_champion_meta.sql",
    "scripts/sql/create_champion_synergy.sql",
    "scripts/sql/create_champion_counters.sql",
    "scripts/sql/create_player_champion_stats.sql",
    "scripts/sql/create_team_performance.sql",
    "scripts/sql/create_team_form.sql"
]:
    print(f"▶ running {f}")
    subprocess.run(f"duckdb {DB} < {f}", shell=True, check=True)

# %% [markdown]
# ## 2) Peek at feature tables

# %%
import duckdb, pandas as pd

con = duckdb.connect(DB)

tables = [
    "team_picks", "match_picks", 
    "champion_meta","champion_synergy","champion_counters",
    "player_champion_stats","team_performance","team_form"
]

for t in tables:
    df = con.execute(f"SELECT * FROM {t} LIMIT 5").df()
    print(f"\n### {t} (5 rows)")
    display(df)

# %% [markdown]
# ## 3) Run training & evaluation for each iteration

# %%
import os, joblib
import numpy as np
from lolpredictor.etl import get_connection, load_mvp_features
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from scripts.train import train_and_save  # noqa: E402
from scripts.evaluate import main as evaluate_main  # noqa: E402
import config

# ensure models dir exists
os.makedirs(config.MODELS_DIR, exist_ok=True)

results = []
for iteration in ["baseline","meta","counters","profiles","form"]:
    out = os.path.join(
        config.MODELS_DIR,
        f"{iteration}_{pd.Timestamp.now().strftime('%Y%m%d')}.pkl"
    )
    print(f"\n## Training: {iteration}")
    train_and_save(iteration, config.DB_PATH, out)

    print(f"\n## Evaluating: {iteration}")
    # capture stdout of evaluate.py by monkey‐patching args
    import sys
    sys.argv = [
        "evaluate.py",
        "--model", out,
        "--db", config.DB_PATH,
        "--split-date", config.SPLIT_DATE
    ]
    evaluate_main()

# %% [markdown]
# ## 4) Summary: manually inspect the printed metrics above,
# or you can wrap the evaluation outputs into a DataFrame for plotting.

# %%
# (Optional) code here to parse the printed logs into a DataFrame
