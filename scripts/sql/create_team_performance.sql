-- FILE: scripts/sql/create_team_performance.sql
DROP TABLE IF EXISTS team_performance;

CREATE TABLE team_performance AS
SELECT
  tp.gameid        AS match_id,
  tp.teamid,
  tp.game_date,

  -- If SUM(killsat15) is NULL (because the data is partial), use 0 instead.
  COALESCE(SUM(rp.killsat15), 0)    AS team_kills15,
  COALESCE(SUM(rp.assistsat15), 0)  AS team_assists15,
  COALESCE(SUM(rp.deathsat15), 0)   AS team_deaths15,

  -- If AVG(golddiffat15) is NULL, use 0.0 instead.
  COALESCE(AVG(rp.csdiffat15), 0.0)   AS avg_csdiff15,
  COALESCE(AVG(rp.golddiffat15), 0.0) AS avg_golddiff15,
  COALESCE(AVG(rp.xpdiffat15), 0.0)   AS avg_xpdiff15,

  tp.label
FROM team_picks tp
JOIN raw_player_stats rp
  ON tp.gameid = rp.gameid
 AND tp.teamid = rp.teamid
WHERE rp.playerid IS NOT NULL
GROUP BY tp.gameid, tp.teamid, tp.game_date, tp.label;