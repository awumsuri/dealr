import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel

# Replace with your actual API key
API_KEY = "YOUR_THE_ODDS_API_KEY"

app = FastAPI()

# ---------- Data Models ---------- #
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

# ---------- Fetch Historical Data ---------- #
def fetch_historical_data(sport_key: str, regions: str = 'us', markets: str = 'h2h', date_format: str = 'iso') -> pd.DataFrame:
    """Fetch historical odds data from The Odds API and return as a DataFrame."""
    # For demonstration, we're querying the 'odds' endpoint (in practice,
    # you'd use your subscription's historical endpoint and store results).
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "dateFormat": date_format
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Minimal processing example:
    # Flatten the odds into a DataFrame. Real usage might need deeper processing.
    rows = []
    for game in data:
        home_team = game.get("home_team")
        away_team = [team for team in game.get("teams", []) if team != home_team]
        away_team = away_team[0] if away_team else None

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    row = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker["title"],
                        "team": outcome["name"],
                        "price": outcome["price"],
                        "commence_time": game["commence_time"]
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df

# ---------- Training a Model ---------- #
def train_model(df: pd.DataFrame):
    """
    Example training function. Assumes df has columns like:
    'home_team', 'away_team', 'team', 'price', 'commence_time'.
    In practice, you'll need actual game results for your 'y' label.
    """
    # Dummy label creation: let's say if the 'team' is the home_team, we label 1, else 0.
    # Replace with real match outcome logic.
    df['label'] = df.apply(lambda row: 1 if row['team'] == row['home_team'] else 0, axis=1)

    # Numeric features example:
    df['home_team_encoded'] = df['home_team'].astype('category').cat.codes
    df['away_team_encoded'] = df['away_team'].astype('category').cat.codes
    df['team_encoded']      = df['team'].astype('category').cat.codes

    X = df[['home_team_encoded', 'away_team_encoded', 'team_encoded', 'price']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

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
@app.on_event("startup")
def startup_event():
    global trained_model
    historical_df = fetch_historical_data("americanfootball_nfl")  # example sport key
    trained_model = train_model(historical_df)

# ---------- API Routes ---------- #
@app.post("/predict")
def get_prediction(pred_req: PredictionRequest):
    global trained_model
    prediction = predict_winner(trained_model, pred_req.home_team, pred_req.away_team)
    return {"predicted_winner": prediction}

@app.get("/")
def health_check():
    return {"status": "ok"}
