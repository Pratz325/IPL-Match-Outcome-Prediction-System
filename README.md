# IPL Match Outcome Prediction System

This project develops an intelligent system to predict Indian Premier League (IPL) cricket match outcomes and key events using machine learning.

## Features

- **Match Winner Predictor**: Random Forest Classifier predicts the winner based on teams, toss decision, venue, and season. Achieves 85% accuracy.
- **Six Predictor**: Logistic Regression estimates probability of at least one six in the first two overs. 72% accuracy, ROC-AUC 0.76.
- **Web Interface**: Flask app for real-time predictions via user-friendly forms.

## Dataset

Sourced from IPL records ([matches.csv](https://www.kaggle.com/manasgarg/ipl), [deliveries.csv](https://www.kaggle.com/manasgarg/ipl)).

- **Features (Winner)**: season, venue, toss_winner, toss_decision, batting_team, bowling_team. 
- **Features (Sixes)**: batting_team, bowling_team.  
- Preprocessing: Missing value imputation, feature engineering (toss-based team assignment), categorical encoding.

## Models

| Model | Algorithm | Key Hyperparams | Performance |
|-------|-----------|-----------------|-------------|
| Match Winner | Random Forest | n_estimators=200, max_depth=20 | 85% accuracy |
| Six Prediction | Logistic Regression | max_iter=2000 | 72% accuracy |

- Trained on 80-20 stratified split (random_state=42).
- Top features: Toss decision, venue.

## Quick Start

1. Clone the repo and install dependencies:
  ``bash
  pip install -r requirements.txt

2. Download IPL dataset from Kaggle into data/ (or retrain models).

3. Run the Flask app:
   python main.py

4. Open http://127.0.0.1:5000 in your browser.

## Project Structure:
ML_PROJECT
├── main.py              # Flask app
├── models/              # Pickled models (ipl_winner_model.pkl, rf_model.pkl, etc.)
├── templates/           # HTML (index.html, six_predictor.html)
├── data/                # IPL CSVs (matches.csv, deliveries.csv)
├── requirements.txt
└── README.md

## Results Highlights
- Random Forest outperformed baselines by 8% accuracy.
- Handles unseen teams via one-hot encoding and default zeros.

## Limitations
- Six prediction limited by team-only features.
- No real-time player stats.

## Future Work
- Add live player metrics.
- Expand to more leagues.
- Mobile deployment.
