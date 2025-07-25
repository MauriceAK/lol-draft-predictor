LoL Esports Win Predictor - Data Pipeline
This project contains a data processing pipeline for League of Legends esports match data. It's designed to take raw match history CSVs from sources like Oracle's Elixir, clean the data, and transform it into a structured format suitable for machine learning analysis to predict game outcomes based on draft information.

Project Structure
The project is organized to separate data, source code, and outputs, which is a best practice for data science projects.

LOL-DRAFT-PROJECT/
├── data/
│   ├── processed/      # Output location for cleaned, analysis-ready data
│   └── raw/            # Place all your raw .csv match history files here
├── src/
│   └── data_processing.py  # Core Python module with data processing functions
├── venv/                 # Virtual environment directory (created by you)
├── .gitignore            # Tells Git which files to ignore (e.g., venv)
├── README.md             # This file
├── requirements.txt      # Lists all Python package dependencies
└── run_pipeline.py       # The main script to execute the entire data pipeline

Setup and Installation
To get this project running on your local machine, follow these steps.

1. Clone the Repository

git clone <your-repository-url>
cd LOL-DRAFT-PROJECT

2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project-specific dependencies.

# Create the environment
python3 -m venv venv

# Activate the environment (for Linux/macOS)
source venv/bin/activate

# For Windows, use:
# venv\Scripts\activate

3. Install Dependencies
Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt

How to Use
1. Add Raw Data
Place all your raw League of Legends match data .csv files into the data/raw/ directory. The pipeline will automatically detect and process all CSV files in this folder.

2. Run the Pipeline
Execute the main script from the project's root directory to start the data processing.

python3 run_pipeline.py

The script will:

Find all .csv files in data/raw/.

Process and aggregate the data into a game-centric format.

Save the final, cleaned dataset as processed_lol_matches.csv in the data/processed/ directory.

Pipeline Details
Input: Raw CSV files containing player-level match data.

Output: A single CSV file where each row represents one game, with columns for:

gameid

winner (Blue or Red)

Blue/Red Team Name

Blue/Red Team Players (list)

Blue/Red Team Champions (list)

Blue/Red Team Bans (list)