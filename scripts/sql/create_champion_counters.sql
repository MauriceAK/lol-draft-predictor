-- Drop old
DROP TABLE IF EXISTS champion_counters;

-- For each champ A vs champ B on opposite teams, how often A’s team wins
CREATE TABLE champion_counters AS
SELECT
  p1.champion AS champ_a,
  p2.champion AS champ_b,
  AVG(CASE WHEN p1.result = 1 AND p2.result = 0 THEN 1.0 ELSE 0.0 END) AS counter_win_rate
FROM raw_player_stats p1
JOIN raw_player_stats p2
  ON p1.gameid = p2.gameid
 AND p1.teamid != p2.teamid
GROUP BY champ_a, champ_b;
