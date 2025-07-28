-- scripts/sql/create_mvp_view.sql

-- Drop the view if it exists to ensure a clean rebuild
DROP VIEW IF EXISTS match_picks_with_opponents;

-- This view creates a single row for each team's perspective in a match,
-- including the opponent's champion picks and player IDs.
CREATE VIEW match_picks_with_opponents AS
WITH match_teams AS (
    SELECT
        gameid,
        MIN(teamid) AS team1_id,
        MAX(teamid) AS team2_id
    FROM raw_player_stats
    GROUP BY gameid
),
game_level_data AS (
    SELECT
        p.gameid,
        p.date AS game_date,
        p.patch,
        p.side,
        p.teamid,
        MAX(CASE WHEN p.position = 'top' THEN p.playerid END) AS player1_id,
        MAX(CASE WHEN p.position = 'jng' THEN p.playerid END) AS player2_id,
        MAX(CASE WHEN p.position = 'mid' THEN p.playerid END) AS player3_id,
        MAX(CASE WHEN p.position = 'bot' THEN p.playerid END) AS player4_id,
        MAX(CASE WHEN p.position = 'sup' THEN p.playerid END) AS player5_id,
        MAX(CASE WHEN p.position = 'top' THEN p.champion END) AS pick1,
        MAX(CASE WHEN p.position = 'jng' THEN p.champion END) AS pick2,
        MAX(CASE WHEN p.position = 'mid' THEN p.champion END) AS pick3,
        MAX(CASE WHEN p.position = 'bot' THEN p.champion END) AS pick4,
        MAX(CASE WHEN p.position = 'sup' THEN p.champion END) AS pick5,
        MAX(p.result) AS label
    FROM raw_player_stats p
    GROUP BY p.gameid, p.date, p.patch, p.side, p.teamid
)
-- Union the perspectives of team1 and team2
SELECT
    g.gameid AS match_id,
    g.game_date,
    g.patch,
    g.side,
    g.teamid,
    mt.team2_id AS opp_teamid,
    g.label,
    g.player1_id, g.player2_id, g.player3_id, g.player4_id, g.player5_id,
    g.pick1, g.pick2, g.pick3, g.pick4, g.pick5
FROM game_level_data g
JOIN match_teams mt ON g.gameid = mt.gameid AND g.teamid = mt.team1_id
UNION ALL
SELECT
    g.gameid AS match_id,
    g.game_date,
    g.patch,
    g.side,
    g.teamid,
    mt.team1_id AS opp_teamid,
    g.label,
    g.player1_id, g.player2_id, g.player3_id, g.player4_id, g.player5_id,
    g.pick1, g.pick2, g.pick3, g.pick4, g.pick5
FROM game_level_data g
JOIN match_teams mt ON g.gameid = mt.gameid AND g.teamid = mt.team2_id;