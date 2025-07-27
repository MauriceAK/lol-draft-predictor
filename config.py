# config.py

import os

# DuckDB file path
DB_PATH = os.getenv("LOL_DB_PATH", "data/lol.duckdb")

# Where we store models
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.getcwd(), "models"))

# Default hold-out split date
SPLIT_DATE = os.getenv("SPLIT_DATE", "2025-06-01")
