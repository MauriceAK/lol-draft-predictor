-- point your ETL at this view
DROP VIEW IF EXISTS mvp_team_features;

CREATE VIEW mvp_team_features AS
SELECT * 
FROM match_picks;
