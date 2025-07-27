-- scripts/create_player_champion_stats.sql

DROP TABLE IF EXISTS player_champion_stats;

CREATE TABLE player_champion_stats AS
SELECT
  playerid   AS player_id,
  champion,
  AVG(result)     AS player_champ_win_rate,
  COUNT(*)        AS player_champ_games,
  AVG(killsat15)  AS avg_kills15,
  AVG(assistsat15)AS avg_assists15,
  AVG(deathsat15) AS avg_deaths15,
  AVG(csdiffat15) AS avg_csdiff15,
  AVG(xpdiffat15) AS avg_xpdiff15
FROM raw_player_stats
WHERE playerid IS NOT NULL
GROUP BY player_id, champion;
