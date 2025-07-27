# scripts/train.py

import fnmatch
import argparse, os, joblib
from lolpredictor.etl import get_connection, load_mvp_features
import config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

FEATURE_GROUPS = {
    "baseline":   {"cat": [], "num": []},
    "meta":       {"cat": [], "num": ["*_win_rate","*_pick_rate"]},
    "counters":   {"extend": "meta", "num": ["team_synergy*","counter*"]},
    "profiles":   {"extend": "counters", "num": ["pick*_player*"]},
    "form":       {"extend": "profiles", "num": [
        "team_kills*","team_assists*","team_deaths*",
        "avg_*_last5"
    ]},
}

def select_columns(df, iteration):
    cfg = FEATURE_GROUPS[iteration]
    # start with baseline picks
    cat = [f"pick{i}" for i in range(1,6)] + [f"opp_pick{i}" for i in range(1,6)]
    num = []
    # recursively extend
    base = cfg.get("extend")
    if base:
        _, base_num = select_columns(df, base)
        num += base_num
    # add this iteration’s numeric patterns
    for pattern in cfg.get("num", []):
        num += [c for c in df.columns if fnmatch.fnmatch(c, pattern)]
    return cat, num

def train_and_save(iteration, out_path):
    con = get_connection(config.DB_PATH)
    df  = load_mvp_features(con)
    cat_cols, num_cols = select_columns(df, iteration)
    X = df[cat_cols + num_cols]
    y = df["label"].values

    preproc = ColumnTransformer([
        ("ohe",   OneHotEncoder(sparse_output=True, handle_unknown="ignore"), cat_cols),
        ("scale", StandardScaler(), num_cols)
    ])
    pipeline = Pipeline([("preproc", preproc), ("model", LGBMClassifier(random_state=42))])
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"✅ Saved {iteration} model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("iteration", choices=FEATURE_GROUPS.keys())
    parser.add_argument("--out", default=os.path.join(config.MODELS_DIR, "model.pkl"))
    args = parser.parse_args()
    train_and_save(args.iteration, args.out)
