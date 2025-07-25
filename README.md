Of course. Here is a concise, updated README.md that accurately reflects your project's current pipeline and outputs.

LoL Esports Win Predictor - Data Pipeline
This project contains a data processing pipeline for League of Legends esports match data. It's designed to take raw match history CSVs, clean the data, and transform it into a numerical feature set suitable for machine learning.

Project Structure
LOL-DRAFT-PREDICTOR/
├── data/
│   ├── processed/      # Output location for all processed files
│   └── raw/            # Place all raw .csv match history files here
├── src/
│   ├── data_processing.py
│   ├── run_pipeline.py
│   └── ... (other scripts)
├── venv/
├── .gitignore
├── README.md
└── requirements.txt
Setup and Installation
Clone the Repository

Bash

git clone <your-repository-url>
cd LOL-DRAFT-PREDICTOR
Create and Activate a Virtual Environment

Bash

# Create the environment
python -m venv venv
# Activate on Linux/macOS
source venv/bin/activate
# Activate on Windows
# venv\Scripts\activate
Install Dependencies

Bash

pip install -r requirements.txt
How to Use
Add Raw Data
Place your raw .csv match history files into the data/raw/ directory.

Run the Pipeline
Execute the main script from the src directory to start processing.

Bash

python src/run_pipeline.py
Pipeline Outputs
The pipeline processes the raw data and generates several files in the data/processed/ directory:

all_regions_processed.csv: An intermediate file where each row is a single game from all available leagues. Data is aggregated but still contains text (e.g., team names, champion lists).

main_regions_processed.csv: Same as above, but filtered to only include major competitive regions (LCK, LPL, LEC, LCS).

all_regions_ml_features.csv: A model-ready dataset for all regions. All data is numerical, containing features like team win rates and one-hot encoded champion picks/bans.

main_regions_ml_features.csv: The final model-ready dataset for main regions only. Use this for training a model focused on top-tier competitive play.