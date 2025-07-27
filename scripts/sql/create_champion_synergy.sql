-- scripts/create_champion_synergy.sql

DROP TABLE IF EXISTS champion_synergy;

CREATE TABLE champion_synergy AS
WITH team_pairs AS (
  SELECT
    LEAST(p1.champion, p2.champion) AS champ1,
    GREATEST(p1.champion, p2.champion) AS champ2,
    (p1.result + p2.result) / 2.0    AS combined_result
  FROM raw_player_stats p1
  JOIN raw_player_stats p2
    ON p1.gameid = p2.gameid
   AND p1.teamid = p2.teamid
   AND p1.champion < p2.champion
)
SELECT
  champ1,
  champ2,
  AVG(combined_result) AS synergy_win_rate
FROM team_pairs
GROUP BY champ1, champ2;
