import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming processed_data.csv is loaded into 'df'
df = pd.read_csv('processed_data.csv')

# Define feature columns as previously specified
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

# Prepare the feature matrix and target vector
X = df[feature_columns].values
y = df['result'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")

# Integrate into prediction function if necessary
def predict_result(xgb_model, single_df):
    # assuming single_df is preprocessed similarly to training data
    X_single = single_df[feature_columns].values
    prediction = xgb_model.predict(X_single)
    return prediction

# Example usage of the predict function after model training
# prediction = predict_result(xgb_model, df_to_predict)