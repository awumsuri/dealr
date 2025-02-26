import numpy as np
from datetime import datetime
from math import pi
from match_scores import MatchScores
from pymongo import MongoClient

import pandas as pd

class FeatureProcessor:
    def __init__(self, team_id_mapping=None):
        # Initialize with existing team ID mapping if available
        # self.team_id_mapping = team_id_mapping or {}
        self.default_team_id = 0  # For unknown teams
        # self.next_new_id = max(self.team_id_mapping.values(), default=0) + 1
        self.scores = MatchScores()
    def process_row_features(self, row):
        """Process a single row of raw data into model-ready features."""
        features = {}

        # Team ID processing with dynamic mapping
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        
        # Get or create team IDs
        features["home_team_id"] = self._get_team_id(home_team)
        features["away_team_id"] = self._get_team_id(away_team)

        # Cyclical time encoding
        commence_time = datetime.fromisoformat(row["commence_time"].replace("Z", ""))
        self._add_cyclical_features(features, commence_time)

        # Odds processing with implied probabilities
        self._process_odds(features, row, home_team, away_team)

        # Encode results (Win, Loss, Draw) as a single label for classification
        outcome_mapping = {"Win": 0.0, "Draw": 1.0, "Loss": 2.0}
        y = int(outcome_mapping.get(row.get("WLD"), 2))  # default to Loss if missing

        # Convert to numpy array in correct order
        return self._features_to_array(features), y

    def _get_team_id(self, team_name) -> int: 
        """Get or create team ID, maintaining consistent mapping."""
        return int(self.scores.get_team_id(team_name))

    def _add_cyclical_features(self, features, dt):
        """Add sine/cosine transformed time features."""
        # Month (0-11 scale for 12 months)
        month = dt.month - 1
        features["month_sin"] = np.sin(2 * pi * month / 12)
        features["month_cos"] = np.cos(2 * pi * month / 12)

        # Day (0-30 scale for 31 days)
        day = dt.day - 1
        features["day_sin"] = np.sin(2 * pi * day / 31)
        features["day_cos"] = np.cos(2 * pi * day / 31)

        # Hour (0-23 scale)
        hour = dt.hour
        features["hour_sin"] = np.sin(2 * pi * hour / 24)
        features["hour_cos"] = np.cos(2 * pi * hour / 24)

    def _process_odds(self, features, row, home_team, away_team):
        """Process odds data with implied probability calculation."""
        # Initialize odds collection
        odds = {
            "home": [],
            "draw": [],
            "away": []
        }

        # Collect odds from available bookmakers
        if "historical_odds" in row and len(row["historical_odds"]) > 0:
            for bookmaker in row["historical_odds"][0].get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            name = outcome["name"].lower()
                            price = outcome["price"]
                            
                            if name == home_team.lower():
                                odds["home"].append(price)
                            elif name == away_team.lower()  :
                                odds["away"].append(price)
                            elif name == "draw":
                                odds["draw"].append(price)

        # Calculate statistics with fallback
        for outcome in ["home", "draw", "away"]:
            key = outcome
            if not odds[key]:
                odds[key] = [1.0]  # Prevent division by zero

            # Basic statistics
            features[f"{key}_odds_mean"] = np.mean(odds[key])
            features[f"{key}_odds_min"] = np.min(odds[key])
            features[f"{key}_odds_max"] = np.max(odds[key])

            # Implied probability
            features[f"{key}_implied"] = 1 / features[f"{key}_odds_mean"]

        # Normalize probabilities
        total_implied = sum(features[f"{k}_implied"] for k in ["home", "draw", "away"])
        for outcome in ["home", "draw", "away"]:
            features[f"{outcome}_prob"] = features[f"{outcome}_implied"] / total_implied

    def _features_to_array(self, features):
        """Convert features dictionary to numpy array in correct order."""
        feature_order = [
            # Team IDs
            "home_team_id", "away_team_id",
            
            # Cyclical time features
            "month_sin", "month_cos", 
            "day_sin", "day_cos",
            "hour_sin", "hour_cos",
            
            # Probability features
            "home_prob", "draw_prob", "away_prob",
            
            # Odds statistics
            "home_odds_mean", "home_odds_min", "home_odds_max",
            "draw_odds_mean", "draw_odds_min", "draw_odds_max",
            "away_odds_mean", "away_odds_min", "away_odds_max"
        ]
        
        return np.array([features[key] for key in feature_order], dtype=np.float32)
    
class FeatureEngineering:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:7777/')
        self.db = self.client['dealr']
        self.collection = self.db['events']      

    def get_processed_data(self):
        X_all = []
        y_all = []
        
        countErrors = 0
        countProcessed = 0
        
        features = FeatureProcessor()

        for doc in self.collection.find():
            try:
                X, y = features.process_row_features(doc)
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
            'month_sin', 'month_cos', 
            'day_sin', 'day_cos',
            'hour_sin', 'hour_cos',
            'home_prob', 'draw_prob', 'away_prob',
            'home_odds_mean', 'home_odds_min', 'home_odds_max',
            'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
            'away_odds_mean', 'away_odds_min', 'away_odds_max'
        ]
       
       X_df = pd.DataFrame(X_data, columns=feature_columns)
       X_df['result'] = y_data
       X_df.to_csv(filename, index=False)
        
       return X_df

# Usage example:
if __name__ == "__main__":
    # Initialize processor (persist this between uses to maintain team mapping)
    processor = FeatureEngineering()
    X, y = processor.get_processed_data()   
    
    print("Processed features shape:", X.shape)
    print("\nFeature Statistics:")
    feature_names = [
        'home_team_id', 'away_team_id',
        'month_sin', 'month_cos', 
        'day_sin', 'day_cos',
        'hour_sin', 'hour_cos',
        'home_prob', 'draw_prob', 'away_prob',
        'home_odds_mean', 'home_odds_min', 'home_odds_max',
        'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
        'away_odds_mean', 'away_odds_min', 'away_odds_max'
    ]
    
    # Print statistics for each feature
    for i, feature in enumerate(feature_names):
        feature_data = X[:, i]
        print(f"{feature:15} | mean: {feature_data.mean():8.3f} | std: {feature_data.std():8.3f} | "
              f"min: {feature_data.min():8.3f} | max: {feature_data.max():8.3f}")
    
    # Print first few samples
    print("\nFirst 5 samples:")
    for i in range(min(5, len(X))):
        print(f"\nSample {i+1}:")
        for j, feature in enumerate(feature_names):
            print(f"{feature:15}: {X[i][j]:8.3f}")
            
    processor.save_to_csv(X, y, 'processed_data.csv')
