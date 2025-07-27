-- scripts/create_team_metrics.sql

DROP TABLE IF EXISTS team_metrics;

CREATE TABLE team_metrics AS
SELECT
  gameid        AS match_id,
  teamid,
  golddiffat15,
  xpdiffat15,
  csdiffat15,
  killsat15,
  assistsat15,
  deathsat15,
  opp_killsat15,
  opp_assistsat15,
  opp_deathsat15,
  gamelength
FROM raw_player_stats
WHERE champion IS NULL;
