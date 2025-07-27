-- scripts/create_match_stats.sql

DROP TABLE IF EXISTS match_picks;

CREATE TABLE match_picks AS
SELECT
  t1.gameid       AS match_id,
  t1.game_date    AS game_date,
  t1.patch        AS patch,
  t1.side         AS side,
  t1.teamid       AS teamid,

  -- Team’s picks, renamed to pick1…pick5
  t1.pick1        AS pick1,
  t1.pick2        AS pick2,
  t1.pick3        AS pick3,
  t1.pick4        AS pick4,
  t1.pick5        AS pick5,

  -- Opponent’s picks, renamed to opp_pick1…opp_pick5
  t2.pick1        AS opp_pick1,
  t2.pick2        AS opp_pick2,
  t2.pick3        AS opp_pick3,
  t2.pick4        AS opp_pick4,
  t2.pick5        AS opp_pick5,

  -- Team’s players, so we can join player–champ stats
  t1.player1_id   AS player1_id,
  t1.player2_id   AS player2_id,
  t1.player3_id   AS player3_id,
  t1.player4_id   AS player4_id,
  t1.player5_id   AS player5_id,

  -- Label: did Team A win?
  CASE WHEN t1.label = 1 THEN 1 ELSE 0 END AS label

FROM team_picks t1
JOIN team_picks t2
  ON t1.gameid = t2.gameid
 AND t1.teamid < t2.teamid;
