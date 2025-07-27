-- scripts/create_team_form.sql
DROP TABLE IF EXISTS team_form;

CREATE TABLE team_form AS
SELECT
  match_id,
  teamid,

  -- last 5‐game win%
  AVG(label) OVER win_wind         AS win_rate_last5,

  -- last 5‐game avg kills/assists/deaths
  AVG(team_kills15)   OVER kills_wind    AS avg_kills15_last5,
  AVG(team_assists15) OVER assists_wind  AS avg_assists15_last5,
  AVG(team_deaths15)  OVER deaths_wind   AS avg_deaths15_last5,

  -- last 5‐game avg diffs
  AVG(avg_csdiff15)   OVER csdiff_wind   AS avg_csdiff15_last5,
  AVG(avg_golddiff15) OVER golddiff_wind AS avg_golddiff15_last5,
  AVG(avg_xpdiff15)   OVER xpdiff_wind   AS avg_xpdiff15_last5

FROM team_performance

WINDOW
  win_wind       AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
  kills_wind     AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
  assists_wind   AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
  deaths_wind    AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
  csdiff_wind    AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
  golddiff_wind  AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
  xpdiff_wind    AS (PARTITION BY teamid ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING);
