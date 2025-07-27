#!/usr/bin/env python3
# scripts/evaluate.py

# python scripts/evaluate.py \
#   --model models/03_profiles_20250725.pkl

import sys, os, argparse, joblib, numpy as np, duckdb
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from lolpredictor.etl import get_connection, load_mvp_features
import config

def main():
    p = argparse.ArgumentParser(description="Evaluate draft-based model")
    p.add_argument("--model", required=True, help="Path to .pkl model")
    p.add_argument("--db",    default=config.DB_PATH, help="DuckDB path")
    p.add_argument(
        "--split-date",
        default=config.SPLIT_DATE,
        help="Hold-out split date (YYYY-MM-DD)"
    )
    args = p.parse_args()

    # 1. Load data & model
    con      = get_connection(args.db)
    df       = load_mvp_features(con).sort_values("game_date")
    pipeline = joblib.load(args.model)

    # 2. Prepare X & y
    cat_cols = [c for c in df.columns if c.startswith("pick") and c in df]
    num_cols = [c for c in df.columns
                if c not in ("match_id","game_date","label") and c not in cat_cols]
    X, y = df[cat_cols + num_cols], df["label"].values

    # 3. Hold-out evaluation
    mask_tr  = df["game_date"] < args.split_date
    X_train, X_test = X[mask_tr], X[~mask_tr]
    y_train, y_test = y[mask_tr], y[~mask_tr]
    pipe = clone(pipeline)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:,1]
    pred  = (proba > 0.5).astype(int)

    print(f"\n=== Hold-out from {args.split_date} ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("AUC:     ", roc_auc_score(y_test, proba))
    print("LogLoss: ", log_loss(y_test, proba))

    # 4. TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=5)
    accs, aucs, losses = [], [], []
    for fold, (ti, vi) in enumerate(tscv.split(X), 1):
        cvp = clone(pipeline)
        cvp.fit(X.iloc[ti], y[ti])
        p_proba = cvp.predict_proba(X.iloc[vi])[:,1]
        p_pred  = (p_proba > 0.5).astype(int)
        accs.append(accuracy_score(y[vi], p_pred))
        aucs.append(roc_auc_score(y[vi], p_proba))
        losses.append(log_loss(y[vi], p_proba))
        print(f"Fold {fold}: Acc={accs[-1]:.4f}, AUC={aucs[-1]:.4f}, LogLoss={losses[-1]:.4f}")

    print(f"\nCV Summary — Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())  # ensure lolpredictor is importable
    main()
