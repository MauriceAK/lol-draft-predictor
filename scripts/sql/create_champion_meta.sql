-- FILE: scripts/sql/create_champion_meta.sql
-- FIX: Added NULLIF to prevent division-by-zero errors.

DROP TABLE IF EXISTS champion_meta;

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
  -- Use NULLIF to avoid division by zero if a patch somehow has 0 picks
  (pc.pick_count::DOUBLE / NULLIF(pt.total_picks, 0)) AS pick_rate
FROM per_champ pc
JOIN patch_totals pt USING (patch);