-- scripts/create_team_performance.sql
DROP TABLE IF EXISTS team_performance;

CREATE TABLE team_performance AS
SELECT
  tp.gameid        AS match_id,
  tp.teamid,
  tp.game_date,

  -- full-game totals at 15' for each team
  SUM(rp.killsat15)    AS team_kills15,
  SUM(rp.assistsat15)  AS team_assists15,
  SUM(rp.deathsat15)   AS team_deaths15,

  -- average 15' diffs per team
  AVG(rp.csdiffat15)   AS avg_csdiff15,
  AVG(rp.golddiffat15) AS avg_golddiff15,
  AVG(rp.xpdiffat15)   AS avg_xpdiff15,

  tp.label
FROM team_picks tp
JOIN raw_player_stats rp
  ON tp.gameid = rp.gameid
 AND tp.teamid = rp.teamid
WHERE rp.playerid IS NOT NULL
GROUP BY tp.gameid, tp.teamid, tp.game_date, tp.label;
