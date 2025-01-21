import numpy as np
from datetime import datetime
from match_scores import MatchScores
import pandas as pd

def process_row_with_min_max_and_y(row):
    # Initialize feature list for X
    features = []
    
    scores = MatchScores()
    
    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    
    home_team_id = scores.get_team_id(home_team)
    away_team_id = scores.get_team_id(away_team)
    # Team indices (example mapping)
    features.append(home_team_id)
    features.append(away_team_id)

    # Time features
    commence_time = datetime.fromisoformat(row["commence_time"].replace("Z", ""))
    features.extend([commence_time.year, commence_time.month, commence_time.day, commence_time.hour])

    # Aggregate odds
    home_odds = []
    draw_odds = []
    away_odds = []
    # In case historical_odds is missing or empty, handle gracefully
    if "historical_odds" in row and len(row["historical_odds"]) > 0:
        # We only process the first entry of historical_odds in this code
        for bookmaker in row["historical_odds"][0].get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":  # Ensure we only process head-to-head odds
                    outcomes = market.get("outcomes", [])
                    for outcome in outcomes:
                        name_lower = outcome["name"].lower()
                        price = outcome["price"]
                        if name_lower == home_team.lower():
                            home_odds.append(price)
                        elif name_lower == away_team.lower():
                            away_odds.append(price)
                        elif name_lower in ["draw", "tie"]:
                            draw_odds.append(price)
    
    # If any list is empty, fill with fallback to avoid errors
    if not home_odds:
        home_odds = [1.0]
    if not draw_odds:
        draw_odds = [1.0]
    if not away_odds:
        away_odds = [1.0]

    # Calculate average, min, max odds for each outcome
    features.extend([
        np.mean(home_odds), np.min(home_odds), np.max(home_odds),  # Home Win
        np.mean(draw_odds), np.min(draw_odds), np.max(draw_odds),  # Draw
        np.mean(away_odds), np.min(away_odds), np.max(away_odds)   # Away Win
    ])
    
    # Add expected winner based on highest probability = lowest odds → or do it via probabilities:
    avg_home = np.mean(home_odds)
    avg_draw = np.mean(draw_odds)
    avg_away = np.mean(away_odds)

    prob_home = 1.0 / avg_home
    prob_draw = 1.0 / avg_draw
    prob_away = 1.0 / avg_away
    
    # We will consistently use argmax(prob_home, prob_draw, prob_away):
    expected_winner = np.argmax([prob_home, prob_draw, prob_away])
    # features.append(float(expected_winner))

    # Encode results (Win, Loss, Draw) as a single label for classification
    outcome_mapping = {"Win": 0.0, "Draw": 1.0, "Loss": 2.0}
    y = float(outcome_mapping.get(row.get("WLD"), 2.0))  # default to Loss if missing

    return np.array(features), y

def process_row_features(row):
    # Same logic, but no y label
    features = []
    
    scores = MatchScores()
    
    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    
    home_team_id = scores.get_team_id(home_team)
    away_team_id = scores.get_team_id(away_team)
    features.append(home_team_id)
    features.append(away_team_id)

    commence_time = datetime.fromisoformat(row["commence_time"].replace("Z", ""))
    features.extend([commence_time.year, commence_time.month, commence_time.day, commence_time.hour])

    home_odds = []
    draw_odds = []
    away_odds = []
    if "historical_odds" in row and len(row["historical_odds"]) > 0:
        for bookmaker in row["historical_odds"][0].get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = market.get("outcomes", [])
                    for outcome in outcomes:
                        name_lower = outcome["name"].lower()
                        price = outcome["price"]
                        if name_lower == home_team.lower():
                            home_odds.append(price)
                        elif name_lower == away_team.lower():
                            away_odds.append(price)
                        elif name_lower in ["draw", "tie"]:
                            draw_odds.append(price)
    
    if not home_odds:
        home_odds = [1.0]
    if not draw_odds:
        draw_odds = [1.0]
    if not away_odds:
        away_odds = [1.0]

    features.extend([
        np.mean(home_odds), np.min(home_odds), np.max(home_odds),
        np.mean(draw_odds), np.min(draw_odds), np.max(draw_odds),
        np.mean(away_odds), np.min(away_odds), np.max(away_odds)
    ])

    # Add expected winner based on highest probability
    avg_home = np.mean(home_odds)
    avg_draw = np.mean(draw_odds)
    avg_away = np.mean(away_odds)
    prob_home = 1.0 / avg_home
    prob_draw = 1.0 / avg_draw
    prob_away = 1.0 / avg_away
    
    expected_winner = np.argmax([prob_home, prob_draw, prob_away])
    #features.append(float(expected_winner))

    return np.array(features)


from pymongo import MongoClient


class FeatureEngineering:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:7777/')
        self.db = self.client['dealr']
        self.collection = self.db['events']

    def process_row_with_min_max_and_y(self, row):
        return process_row_with_min_max_and_y(row)

    def get_processed_data(self):
        X_all = []
        y_all = []
        
        countErrors = 0
        countProcessed = 0

        for doc in self.collection.find():
            try:
                X, y = self.process_row_with_min_max_and_y(doc)
                X_all.append(X)
                y_all.append(y)
                countProcessed += 1
                print(f"Processed document {countProcessed}")
            except Exception as e:
                countErrors += 1
                print(f"Error processing document {countErrors}: {e}")
                continue

        return np.array(X_all), np.array(y_all)

    def save_to_csv(self, X_data, y_data, filename='processed_data.csv'):
        feature_columns = [
            'home_team_id', 'away_team_id', 
            'year', 'month', 'day', 'hour',
            'home_odds_mean', 'home_odds_min', 'home_odds_max',
            'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
            'away_odds_mean', 'away_odds_min', 'away_odds_max'
           
        ]
        
        X_df = pd.DataFrame(X_data, columns=feature_columns)
        X_df['result'] = y_data
        X_df.to_csv(filename, index=False)
        return X_df

class TestFeatures:
    def __init__(self):
        self.feature_engineering = FeatureEngineering()

    def run_test(self):
        X_data, y_data = self.feature_engineering.get_processed_data()
        self.feature_engineering.save_to_csv(X_data, y_data)
        print(f"Total samples processed: {len(X_data)}")
        print(f"Feature vector shape: {X_data.shape}")
        print(f"Label vector shape: {y_data.shape}")

if __name__ == "__main__":
    test = TestFeatures()
    test.run_test()
