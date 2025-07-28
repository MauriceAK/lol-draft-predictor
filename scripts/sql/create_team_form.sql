-- FILE: scripts/sql/create_team_form.sql
-- FIX 1: Changed window from 5 to 10 preceding games.
-- FIX 2: Used COALESCE to provide default values for teams with no prior games, preventing NULLs/NaNs.

DROP TABLE IF EXISTS team_form;

CREATE TABLE team_form AS
SELECT
  match_id,
  teamid,
  game_date,

  -- Use COALESCE to handle cases where there are no preceding games.
  -- Default win rate to 0.5 (neutral) and others to 0.0.
  COALESCE(AVG(label) OVER win_wind, 0.5) AS win_rate_last10,
  COALESCE(AVG(team_kills15) OVER form_wind, 0.0) AS avg_kills15_last10,
  COALESCE(AVG(team_assists15) OVER form_wind, 0.0) AS avg_assists15_last10,
  COALESCE(AVG(team_deaths15) OVER form_wind, 0.0) AS avg_deaths15_last10,
  COALESCE(AVG(avg_csdiff15) OVER form_wind, 0.0) AS avg_csdiff15_last10,
  COALESCE(AVG(avg_golddiff15) OVER form_wind, 0.0) AS avg_golddiff15_last10,
  COALESCE(AVG(avg_xpdiff15) OVER form_wind, 0.0) AS avg_xpdiff15_last10

FROM team_performance

-- Define windows for a 10-game lookback
WINDOW
  win_wind AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
  form_wind AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);