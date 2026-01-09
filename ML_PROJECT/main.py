from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load Match Winner Prediction Model
with open('models/ipl_winner_model.pkl', 'rb') as model_file:
    winner_model = pickle.load(model_file)
with open('models/feature_columns.pkl', 'rb') as file:
    winner_feature_columns = pickle.load(file)

# Load Six Prediction Model
with open("models/rf_model.pkl", "rb") as six_model_file:
    six_model = pickle.load(six_model_file)
with open("models/encoders.pkl", "rb") as encoder_file:
    encoders = pickle.load(encoder_file)

# Define teams and venues
teams = ["Chennai Super Kings", "Gujrat Titans", "Delhi Capitals", "Kolkata Knight Riders", "Mumbai Indians",
         "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", "Sunrisers Hyderabad"]

venues = ["Wankhede Stadium", "Eden Gardens", "M. Chinnaswamy Stadium", "Feroz Shah Kotla Ground",
          "MA Chidambaram Stadium", "Rajiv Gandhi International Stadium", "Sawai Mansingh Stadium",
          "Punjab Cricket Association Stadium", "Dubai International Cricket Stadium"]

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home route for IPL match winner prediction"""
    winner = None
    if request.method == 'POST':
        custom_input = {
            'batting_team': request.form['batting_team'],
            'bowling_team': request.form['bowling_team'],
            'toss_winner': request.form['toss_winner'],
            'toss_decision': request.form['toss_decision'],
            'venue': request.form['venue'],
            'season': 2025,  # Default season
            'city': request.form['venue']  # Assuming city same as venue
        }

        # Preprocess input
        custom_df = pd.DataFrame([custom_input])
        custom_df['toss_decision'] = custom_df['toss_decision'].map({'bat': 0, 'ball': 1})
        custom_df = pd.get_dummies(custom_df, columns=['batting_team', 'bowling_team', 'toss_winner', 'venue', 'city'])

        # Ensure input matches training feature columns
        for col in winner_feature_columns:
            if col not in custom_df:
                custom_df[col] = 0  # Add missing columns
        custom_df = custom_df[winner_feature_columns]

        prediction = winner_model.predict(custom_df)
        winner = request.form['batting_team'] if prediction[0] == 1 else request.form['bowling_team']

    return render_template('index.html', teams=teams, venues=venues, winner=winner)

@app.route('/six-predictor', methods=['GET', 'POST'])
def six_predictor():
    """Route for six prediction in first 2 overs"""
    prediction_result = None
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']

        if batting_team in encoders['batting_team'].classes_ and bowling_team in encoders['bowling_team'].classes_:
            custom_input = np.array([
                [
                    encoders['batting_team'].transform([batting_team])[0],
                    encoders['bowling_team'].transform([bowling_team])[0]
                ]
            ])
            custom_proba = six_model.predict_proba(custom_input)[:, 1][0]
            predicted_class = 1 if custom_proba > 0.5 else 0

            prediction_result = f"Probability of at least one six: {custom_proba:.4f}. Predicted: {'Yes' if predicted_class == 1 else 'No'}"
        else:
            prediction_result = "Error: One or both teams are not in the training data."

    return render_template('six_predictor.html', teams=teams, prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
