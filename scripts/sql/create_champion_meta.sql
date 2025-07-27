-- scripts/create_champion_meta.sql

-- 1) Drop old table if it exists
DROP TABLE IF EXISTS champion_meta;

-- 2) For each patch & champion, compute:
--    • win_rate  = avg(result)
--    • pick_rate = count(*) / total_picks_in_patch
CREATE TABLE champion_meta AS
WITH per_champ AS (
  SELECT
    patch,
    champion,
    AVG(result)              AS win_rate,
    COUNT(*)                 AS pick_count
  FROM raw_player_stats
  GROUP BY patch, champion
),
patch_totals AS (
  SELECT
    patch,
    SUM(pick_count)          AS total_picks
  FROM per_champ
  GROUP BY patch
)
SELECT
  pc.patch,
  pc.champion,
  pc.win_rate,
  (pc.pick_count::DOUBLE / pt.total_picks) AS pick_rate
FROM per_champ pc
JOIN patch_totals pt USING (patch);
