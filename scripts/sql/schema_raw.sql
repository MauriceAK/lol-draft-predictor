-- drop old, then load CSV
DROP TABLE IF EXISTS raw_player_stats;

CREATE TABLE raw_player_stats AS
SELECT *
FROM read_csv_auto('data/2025_LoL_esports_match_data_from_OraclesElixir.csv');

INSERT INTO raw_player_stats
SELECT *
FROM read_csv_auto('data/2024_LoL_esports_match_data_from_OraclesElixir.csv');

INSERT INTO raw_player_stats
SELECT *
FROM read_csv_auto('data/2023_LoL_esports_match_data_from_OraclesElixir.csv');
