# **LoL Esports Win Predictor - Data & Model Pipeline**

This project contains a full data-to-model pipeline for League of Legends esports. It processes raw match history, engineers features suitable for machine learning, and trains a predictive model.

## **Project Structure**

```
LOL-DRAFT-PREDICTOR/
├── data/
│   ├── processed/
│   └── raw/
├── models/             # Output location for trained model files
├── src/
│   ├── data_processing.py
│   ├── run_pipeline.py     # Main script for data processing
│   ├── train_model.py      # Main script for model training
│   └── predict.py        # Script for making predictions
├── venv/
├── .gitignore
├── README.md
└── requirements.txt
```

## **Setup**

1.  **Clone the Repository** and navigate into the directory.
2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## **How to Use**

The project is now a two-step process: first process the data, then train the model.

### **Step 1: Process the Data**

1.  Place your raw `.csv` match history files into the `data/raw/` directory.
2.  Run the data pipeline script. This will generate the `ml_features.csv` files in `data/processed/`.

    ```bash
    python src/run_pipeline.py
    ```

### **Step 2: Train the Model**

Once the data is processed, run the training script. This will use the processed data to train an XGBoost model and save it in the `models/` directory.

```bash
python src/train_model.py
```

### **Step 3: Make a Prediction**

You can use the `predict.py` script to load the latest model and make a prediction on a sample game.

```bash
python src/predict.py
```