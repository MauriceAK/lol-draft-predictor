-- FILE: scripts/sql/create_champion_counters.sql
-- FIX: Added a HAVING clause to only include counters with at least 5 games played.

DROP TABLE IF EXISTS champion_counters;

CREATE TABLE champion_counters AS
SELECT
  p1.champion AS champ_a,
  p2.champion AS champ_b,
  AVG(CASE WHEN p1.result = 1 THEN 1.0 ELSE 0.0 END) AS counter_win_rate,
  COUNT(*) AS games_played
FROM raw_player_stats p1
JOIN raw_player_stats p2
  ON p1.gameid = p2.gameid
 AND p1.teamid != p2.teamid
GROUP BY champ_a, champ_b
-- Only include matchups that have occurred a reasonable number of times
HAVING COUNT(*) >= 5;