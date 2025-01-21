import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ---------- Data Models ---------- #
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

# ---------- Training a Model ---------- #
def train_model(df: pd.DataFrame):
    """
    Example training function. Assumes df has columns like:
    'home_team', 'away_team', 'team', 'price', 'commence_time'.
    In practice, you'll need actual game results for your 'y' label.
    """
    

    X = df[[
        'home_team_id', 'away_team_id', 'year', 'month', 'day', 'hour',
        'home_odds_mean', 'home_odds_min', 'home_odds_max',
        'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
        'away_odds_mean', 'away_odds_min', 'away_odds_max'
    ]]
    y = df['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Print model accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
    
    # Print probability predictions for test set
    test_probabilities = model.predict_proba(X_test)
    print("\nSample probability predictions (first 5 test cases):")
    for i in range(min(5, len(X_test))):
        print(f"Case {i+1}: Class 0: {test_probabilities[i][0]:.2f}, Class 1: {test_probabilities[i][1]:.2f}")

    return model

# ---------- Prediction Logic ---------- #
def predict_winner(model, home_team: str, away_team: str) -> str:
    """Predict which team is favored (home or away)."""
    # Encode input
    data = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'team': [home_team],  # We'll do a separate check for away
        'price': [1.5]        # Dummy example price; replace with real logic
    })
    data['home_team_encoded'] = data['home_team'].astype('category').cat.codes
    data['away_team_encoded'] = data['away_team'].astype('category').cat.codes
    data['team_encoded']      = data['team'].astype('category').cat.codes

    home_probability = model.predict_proba(data[['home_team_encoded','away_team_encoded','team_encoded','price']])[0][1]

    # Do the same for the away side:
    data_away = data.copy()
    data_away['team'] = away_team
    data_away['team_encoded'] = data_away['team'].astype('category').cat.codes
    away_probability = model.predict_proba(data_away[['home_team_encoded','away_team_encoded','team_encoded','price']])[0][1]

    return home_team if home_probability > away_probability else away_team

# ---------- Build & Train a Model at Startup ---------- #
# In a real production environment, you might load a pre-trained model from disk
# or periodically update the model with a scheduled job.

