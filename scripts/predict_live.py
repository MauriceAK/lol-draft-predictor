#!/usr/bin/env python3
# scripts/predict_live.py

import argparse
import datetime
import json
import joblib
import os
import pandas as pd

import config
from lolpredictor.etl import (
    load_champion_meta,
    load_champion_synergy,
    load_champion_counters,
    load_player_champ_stats,
    load_team_performance,
    load_team_form,
    transform_team_match_rows,
    get_connection,
)

def predict_from_dict(draft: dict, model_path: str):
    """
    draft JSON must include:
      - patch       : e.g. "13.15"
      - blue_team   : org name, e.g. "T1"
      - red_team    : org name, e.g. "GenG"
      - blue_picks  : list of 5 champ strings
      - red_picks   : list of 5 champ strings
    """
    # 1) Map team names → teamid strings
    mapping_path = os.path.join("scripts", "team_mapping.json")
    with open(mapping_path) as mf:
        name2id = json.load(mf)

    blue_id = name2id.get(draft["blue_team"])
    red_id  = name2id.get(draft["red_team"])
    if blue_id is None:
        raise KeyError(f"Unknown blue_team '{draft['blue_team']}' in team_mapping.json")
    if red_id is None:
        raise KeyError(f"Unknown red_team '{draft['red_team']}' in team_mapping.json")

    blue_picks = draft["blue_picks"]
    red_picks  = draft["red_picks"]

    # 2) Champion‐name validation
    con   = get_connection(config.DB_PATH)
    meta  = load_champion_meta(con)
    valid = set(meta["champion"])
    bad   = [c for c in blue_picks + red_picks if c not in valid]
    if bad:
        raise ValueError(
            "Invalid champion name(s) in draft JSON: "
            + ", ".join(bad)
            + "\nValid champions are:\n"
            + ", ".join(sorted(valid))
        )

    # 3) Determine the “t1” side by lex order on teamid
    if blue_id < red_id:
        t1_id, t1_side, t1_picks, t2_picks = blue_id, "blue", blue_picks, red_picks
        blue_is_t1 = True
    else:
        t1_id, t1_side, t1_picks, t2_picks = red_id, "red", red_picks, blue_picks
        blue_is_t1 = False

    # 4) Build a single raw row for transform_team_match_rows
    today = datetime.date.today().isoformat()
    row = {
        "match_id":  0,
        "game_date": today,
        "patch":     float(draft["patch"]),
        "side":      t1_side,
        "teamid":    t1_id,
        "label":     None
    }
    # own picks
    for i, champ in enumerate(t1_picks, start=1):
        row[f"pick{i}"] = champ
    # opponent picks
    for i, champ in enumerate(t2_picks, start=1):
        row[f"opp_pick{i}"] = champ

    roles = ["top","jng","mid","bot","sup"]
    for i, role in enumerate(roles, start=1):
        q = """
        SELECT playerid
            FROM raw_player_stats
        WHERE teamid    = ?
            AND position  = ?
        ORDER BY date DESC
        LIMIT 1
        """
        res = con.execute(q, [t1_id, role]).fetchone()
        row[f"player{i}_id"] = res[0] if res is not None else None
    
    # no player IDs at live time
    # for i in range(1,6):
    #     row[f"player{i}_id"] = None
    print(row)
    #exit()
    raw_df = pd.DataFrame([row])

    # 5) Load all feature tables
    syn  = load_champion_synergy(con)
    ctr  = load_champion_counters(con)
    pc   = load_player_champ_stats(con)
    perf = load_team_performance(con)
    form = load_team_form(con)

    # 6) Transform into full feature set
    feat_df = transform_team_match_rows(raw_df, meta, syn, ctr, pc, perf, form)
    feat_df = feat_df.reset_index(drop=True)

    # 7) Prepare X_live exactly as during training
    cat_cols = [f"pick{i}" for i in range(1,6)] + [f"opp_pick{i}" for i in range(1,6)]
    num_cols = [
        c for c in feat_df.columns
        if c not in ("match_id","game_date","label") and c not in cat_cols
    ]
    X_live = feat_df[cat_cols + num_cols]

    # —— DEBUG: before alignment
    print("🔍 Live features before alignment:", X_live.shape)

    # 8) Load model
    model = joblib.load(model_path)
    print(f"🔍 Loaded model from: {model_path}")

    # 9) Align to training feature order
    feat_order = getattr(model, "feature_names_in_", X_live.columns.tolist())
    X_live = X_live[feat_order]

    # —— DEBUG: after alignment
    print("🔍 Live features after alignment:", X_live.shape)
    print(X_live.T)

    # 10) Predict
    p_t1 = model.predict_proba(X_live)[:,1][0]

    # 11) Map back to blue/red probabilities
    if blue_is_t1:
        p_blue, p_red = p_t1, 1 - p_t1
    else:
        p_blue, p_red = 1 - p_t1, p_t1

    print()
    print(f"Blue side  ({draft['blue_team']}) win probability: {p_blue:.1%}")
    print(f"Red side   ({draft['red_team']}) win probability: {p_red:.1%}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Live predict after draft-lock via JSON draft file."
    )
    parser.add_argument(
        "--draft-file", required=True,
        help="Path to draft JSON (see scripts/draft_template.json)"
    )
    parser.add_argument(
        "--model",
        default=os.path.join(config.MODELS_DIR, "model.pkl"),
        help="Path to trained .pkl model"
    )
    args = parser.parse_args()

    with open(args.draft_file) as f:
        draft = json.load(f)

    # ensure all required keys
    for key in ("patch","blue_team","red_team","blue_picks","red_picks"):
        if key not in draft:
            parser.error(f"Draft JSON missing required key: '{key}'")

    predict_from_dict(draft, args.model)

if __name__ == "__main__":
    main()
