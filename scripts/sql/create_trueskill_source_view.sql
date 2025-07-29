-- This script creates a clean, ordered view of game results,
-- which will be the source data for the Python TrueSkill calculation.

DROP VIEW IF EXISTS game_results_for_trueskill;

CREATE VIEW game_results_for_trueskill AS
SELECT
    gameid AS match_id,
    date AS game_date,
    league,
    MAX(CASE WHEN result = 1 THEN teamid END) AS winner_id,
    MAX(CASE WHEN result = 0 THEN teamid END) AS loser_id
FROM raw_player_stats
WHERE
    gameid IS NOT NULL
    AND teamid IS NOT NULL
GROUP BY
    gameid, date, league
HAVING
    -- Use HAVING to filter after aggregation.
    -- This ensures the game had a clear winner and loser.
    winner_id IS NOT NULL
    AND loser_id IS NOT NULL
ORDER BY
    date, gameid;