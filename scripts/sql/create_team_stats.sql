-- scripts/create_team_stats.sql

DROP TABLE IF EXISTS team_picks;

CREATE TABLE team_picks AS
WITH numbered AS (
  SELECT
    gameid,
    date        AS game_date,
    patch,
    side,
    teamid,
    participantid,
    champion,
    result,
    ROW_NUMBER() OVER (
      PARTITION BY gameid, teamid
      ORDER BY participantid
    ) AS slot
  FROM raw_player_stats
  WHERE playerid IS NOT NULL
)
SELECT
  gameid,
  game_date,
  patch,
  side,
  teamid,
  -- Pivot champions
  MAX(CASE WHEN slot=1 THEN champion END)       AS pick1,
  MAX(CASE WHEN slot=2 THEN champion END)       AS pick2,
  MAX(CASE WHEN slot=3 THEN champion END)       AS pick3,
  MAX(CASE WHEN slot=4 THEN champion END)       AS pick4,
  MAX(CASE WHEN slot=5 THEN champion END)       AS pick5,
  -- Pivot participant IDs
  MAX(CASE WHEN slot=1 THEN participantid END)  AS player1_id,
  MAX(CASE WHEN slot=2 THEN participantid END)  AS player2_id,
  MAX(CASE WHEN slot=3 THEN participantid END)  AS player3_id,
  MAX(CASE WHEN slot=4 THEN participantid END)  AS player4_id,
  MAX(CASE WHEN slot=5 THEN participantid END)  AS player5_id,
  -- Label (did this team win?)
  MAX(result)                                  AS label
FROM numbered
GROUP BY gameid, game_date, patch, side, teamid;
