import torch
import torch.nn as nn
import torch.optim as optim
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Replace with your actual API key
API_KEY = "YOUR_THE_ODDS_API_KEY"

app = FastAPI()

# ---------- Data Models ---------- #
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

# ---------- Neural Network Model ---------- #
class BettingNN(nn.Module):
    def __init__(self, input_size):
        super(BettingNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# ---------- Fetch Historical Data ---------- #
def fetch_historical_data(sport_key: str, regions: str = 'us', markets: str = 'h2h', date_format: str = 'iso') -> pd.DataFrame:
    """Fetch historical odds data from The Odds API and return as a DataFrame."""
    url = f'https://api.sportsgameodds.com/v1/events?leagueID=NFL&startsAfter={startsAfter}&startsBefore={startsBefore}'
    headers = {'x-api-key': os.getenv('SPORTS_GAME_ODDS_API_KEY')}

    all_events = []
    next_cursor = None

    while True:
        try:
            response = requests.get(url, params={'cursor': next_cursor}, headers = headers)
            json = response.json()
            all_events.extend(json['data'])
            next_cursor = json.get('nextCursor')
            if not next_cursor:
                break
        except Exception as e:
            print(f'Error fetching NFL odds: {e}')
            break

    return all_events
    # [Previous fetch_historical_data implementation remains the same]
    # ... existing code ...

# ---------- Training a Model ---------- #
def train_model(df: pd.DataFrame):
    """Train a PyTorch neural network model."""
    # Prepare the data
    df['label'] = df.apply(lambda row: 1 if row['team'] == row['home_team'] else 0, axis=1)
    df['home_team_encoded'] = df['home_team'].astype('category').cat.codes
    df['away_team_encoded'] = df['away_team'].astype('category').cat.codes
    df['team_encoded'] = df['team'].astype('category').cat.codes

    X = df[['home_team_encoded', 'away_team_encoded', 'team_encoded', 'price']]
    y = df['label']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test.values).reshape(-1, 1)

    # Initialize model, loss function, and optimizer
    model = BettingNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    batch_size = 32
    n_batches = len(X_train) // batch_size

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/n_batches:.4f}')

    return model, scaler

# ---------- Prediction Logic ---------- #
def predict_winner(model, scaler, home_team: str, away_team: str) -> str:
    """Predict which team is favored using the PyTorch model."""
    model.eval()
    with torch.no_grad():
        # Prepare input data
        data = pd.DataFrame({
            'home_team': [home_team],
            'away_team': [away_team],
            'team': [home_team],
            'price': [1.5]
        })
        
        data['home_team_encoded'] = data['home_team'].astype('category').cat.codes
        data['away_team_encoded'] = data['away_team'].astype('category').cat.codes
        data['team_encoded'] = data['team'].astype('category').cat.codes

        X = data[['home_team_encoded', 'away_team_encoded', 'team_encoded', 'price']]
        X_scaled = scaler.transform(X)
        home_tensor = torch.FloatTensor(X_scaled)
        home_probability = model(home_tensor).item()

        # Prepare away team prediction
        data_away = data.copy()
        data_away['team'] = away_team
        data_away['team_encoded'] = data_away['team'].astype('category').cat.codes
        X_away = data_away[['home_team_encoded', 'away_team_encoded', 'team_encoded', 'price']]
        X_away_scaled = scaler.transform(X_away)
        away_tensor = torch.FloatTensor(X_away_scaled)
        away_probability = model(away_tensor).item()

    return home_team if home_probability > away_probability else away_team

# ---------- Build & Train a Model at Startup ---------- #
@app.on_event("startup")
def startup_event():
    global trained_model, scaler
    historical_df = fetch_historical_data("americanfootball_nfl")
    trained_model, scaler = train_model(historical_df)

# ---------- API Routes ---------- #
@app.post("/predict")
def get_prediction(pred_req: PredictionRequest):
    global trained_model, scaler
    prediction = predict_winner(trained_model, scaler, pred_req.home_team, pred_req.away_team)
    return {"predicted_winner": prediction}

@app.get("/")
def health_check():
    return {"status": "ok"}
