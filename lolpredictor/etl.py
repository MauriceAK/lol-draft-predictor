# lolpredictor/etl.py

import duckdb
import pandas as pd

def get_connection(db_path: str = "data/lol.duckdb"):
    """Return a DuckDB connection to the given database file."""
    return duckdb.connect(db_path)

def load_team_match_rows(con):
    """Pull raw head-to-head rows (match_picks)."""
    return con.execute("SELECT * FROM match_picks").df()

def load_champion_meta(con):
    return con.execute("""
        SELECT patch, champion, win_rate, pick_rate
        FROM champion_meta
    """).df()

def load_champion_synergy(con):
    return con.execute("""
        SELECT champ1, champ2, synergy_win_rate
        FROM champion_synergy
    """).df()

def load_champion_counters(con):
    return con.execute("""
        SELECT champ_a, champ_b, counter_win_rate
        FROM champion_counters
    """).df()

def load_player_champ_stats(con):
    return con.execute("""
        SELECT player_id, champion,
               player_champ_win_rate, player_champ_games,
               avg_kills15, avg_assists15, avg_deaths15,
               avg_csdiff15, avg_xpdiff15
        FROM player_champion_stats
    """).df()

def load_team_performance(con):
    return con.execute("""
        SELECT match_id, teamid, game_date,
               team_kills15, team_assists15, team_deaths15,
               avg_csdiff15, avg_golddiff15, avg_xpdiff15
        FROM team_performance
    """).df()

def load_team_form(con):
    return con.execute("""
        SELECT match_id, teamid,
               win_rate_last5,
               avg_kills15_last5, avg_assists15_last5, avg_deaths15_last5,
               avg_csdiff15_last5, avg_golddiff15_last5, avg_xpdiff15_last5
        FROM team_form
    """).df()

def transform_team_match_rows(raw_df, meta_df, syn_df, ctr_df, pc_df, perf_df, form_df):
    """
    Merge raw match rows with all feature tables.
    For live (match_id==0) defaults, uses each team's latest performance & form.
    """
    # 1) Build lookup maps for historical matches
    win_map   = {(r.patch, r.champion): r.win_rate    for r in meta_df.itertuples()}
    pick_map  = {(r.patch, r.champion): r.pick_rate   for r in meta_df.itertuples()}
    syn_map   = {(r.champ1, r.champ2): r.synergy_win_rate for r in syn_df.itertuples()}
    ctr_map   = {(r.champ_a, r.champ_b): r.counter_win_rate for r in ctr_df.itertuples()}
    pc_map    = {(r.player_id, r.champion): r for r in pc_df.itertuples()}
    perf_map  = {(r.match_id, r.teamid): r for r in perf_df.itertuples()}
    form_map  = {(r.match_id, r.teamid): r for r in form_df.itertuples()}

    # 2) Build "latest" maps for live prediction
    #    Pick the most recent game_date per team
    perf_latest = {}
    for r in perf_df.itertuples():
        tid = r.teamid
        if tid not in perf_latest or r.game_date > perf_latest[tid].game_date:
            perf_latest[tid] = r
    # Map that to form rows as well
    form_latest = {}
    for tid, pr in perf_latest.items():
        fr = form_map.get((pr.match_id, tid))
        if fr:
            form_latest[tid] = fr

    records = []
    for row in raw_df.itertuples():
        mid    = row.match_id
        teamid = row.teamid
        patch  = row.patch

        # Picks & player IDs
        picks   = [row.pick1, row.pick2, row.pick3, row.pick4, row.pick5]
        opps    = [row.opp_pick1, row.opp_pick2, row.opp_pick3, row.opp_pick4, row.opp_pick5]
        players = [row.player1_id, row.player2_id, row.player3_id, row.player4_id, row.player5_id]

        feat = {}

        # --- Champion meta rates ---
        for i, champ in enumerate(picks, 1):
            feat[f"pick{i}_win_rate"]  = win_map.get((patch, champ), 0.5)
            feat[f"pick{i}_pick_rate"] = pick_map.get((patch, champ), 0.0)
        for i, champ in enumerate(opps, 1):
            feat[f"opp_pick{i}_win_rate"]  = win_map.get((patch, champ), 0.5)
            feat[f"opp_pick{i}_pick_rate"] = pick_map.get((patch, champ), 0.0)

        # --- Team synergy & counter aggregates ---
        syn_vals, ctr_vals = [], []
        for i in range(5):
            for j in range(i+1, 5):
                c1, c2 = picks[i], picks[j]
                syn_vals.append(syn_map.get(tuple(sorted((c1, c2))), 0.5) if c1 and c2 else 0.5)
        for p in picks:
            for o in opps:
                ctr_vals.append(ctr_map.get((p, o), 0.5) if p and o else 0.5)
        feat["team_synergy_mean"] = sum(syn_vals) / len(syn_vals)
        feat["team_synergy_max"]  = max(syn_vals)
        feat["counter_mean"]      = sum(ctr_vals) / len(ctr_vals)
        feat["counter_max"]       = max(ctr_vals)

        # --- Player–Champion profiles ---
        for i, (pid, champ) in enumerate(zip(players, picks), 1):
            stats = pc_map.get((pid, champ))
            if stats:
                feat[f"pick{i}_player_wr"]     = stats.player_champ_win_rate
                feat[f"pick{i}_player_games"]  = stats.player_champ_games
                feat[f"pick{i}_avg_kills15"]   = stats.avg_kills15
                feat[f"pick{i}_avg_assists15"] = stats.avg_assists15
                feat[f"pick{i}_avg_deaths15"]  = stats.avg_deaths15
                feat[f"pick{i}_avg_csdiff15"]  = stats.avg_csdiff15
                feat[f"pick{i}_avg_xpdiff15"]  = stats.avg_xpdiff15
            else:
                feat.update({
                    f"pick{i}_player_wr":      0.5,
                    f"pick{i}_player_games":   0,
                    f"pick{i}_avg_kills15":    0.0,
                    f"pick{i}_avg_assists15":  0.0,
                    f"pick{i}_avg_deaths15":   0.0,
                    f"pick{i}_avg_csdiff15":   0.0,
                    f"pick{i}_avg_xpdiff15":   0.0
                })

        # --- Team performance (either historical or latest) ---
        if mid and (mid, teamid) in perf_map:
            perf = perf_map[(mid, teamid)]
        else:
            perf = perf_latest.get(teamid)
        if perf:
            feat.update({
                "team_kills15":    perf.team_kills15,
                "team_assists15":  perf.team_assists15,
                "team_deaths15":   perf.team_deaths15,
                "avg_csdiff15":    perf.avg_csdiff15,
                "avg_golddiff15":  perf.avg_golddiff15,
                "avg_xpdiff15":    perf.avg_xpdiff15
            })
        else:
            for c in ["team_kills15","team_assists15","team_deaths15",
                      "avg_csdiff15","avg_golddiff15","avg_xpdiff15"]:
                feat[c] = 0.0

        # --- Rolling form (historical or latest) ---
        if mid and (mid, teamid) in form_map:
            frm = form_map[(mid, teamid)]
        else:
            frm = form_latest.get(teamid)
        if frm:
            feat.update({
                "win_rate_last5":       frm.win_rate_last5      or 0.5,
                "avg_kills15_last5":    frm.avg_kills15_last5   or 0.0,
                "avg_assists15_last5":  frm.avg_assists15_last5 or 0.0,
                "avg_deaths15_last5":   frm.avg_deaths15_last5  or 0.0,
                "avg_csdiff15_last5":   frm.avg_csdiff15_last5  or 0.0,
                "avg_golddiff15_last5": frm.avg_golddiff15_last5 or 0.0,
                "avg_xpdiff15_last5":   frm.avg_xpdiff15_last5  or 0.0
            })
        else:
            for c in ["win_rate_last5","avg_kills15_last5","avg_assists15_last5",
                      "avg_deaths15_last5","avg_csdiff15_last5",
                      "avg_golddiff15_last5","avg_xpdiff15_last5"]:
                feat[c] = 0.0

        # --- Label (for historical), default 0 for live ---
        raw_lbl = getattr(row, "label", None)
        label   = int(raw_lbl) if raw_lbl is not None else 0

        # --- Assemble record ---
        rec = {
            "match_id":  mid,
            "game_date": row.game_date,
            **{f"pick{i}":    picks[i-1] for i in range(1,6)},
            **{f"opp_pick{i}": opps[i-1]  for i in range(1,6)},
            **feat,
            "label":     label
        }
        records.append(rec)

    return pd.DataFrame(records)

def load_mvp_features(con):
    """Load and merge all feature tables into the final DataFrame."""
    raw  = load_team_match_rows(con)
    meta = load_champion_meta(con)
    syn  = load_champion_synergy(con)
    ctr  = load_champion_counters(con)
    pc   = load_player_champ_stats(con)
    perf = load_team_performance(con)
    form = load_team_form(con)
    return transform_team_match_rows(raw, meta, syn, ctr, pc, perf, form)
