import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BettingNN(nn.Module):
    def __init__(self, input_size):
        super(BettingNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

def train_model(df: pd.DataFrame):
    """Train a PyTorch neural network model."""
    # Select all features except the target variable 'result'
    X = df[[
        'home_team_id', 'away_team_id', 'year', 'month', 'day', 'hour',
        'home_odds', 'home_odds_min', 'home_odds_max',
        'draw_odds', 'draw_odds_min', 'draw_odds_max',
        'away_odds', 'away_odds_min', 'away_odds_max',
        'home_score', 'away_score'
    ]]
    y = df['result']  # target variable is already encoded as 0,1,2

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train.values)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test.values)

    # Initialize model, loss function, and optimizer
    model = BettingNN(input_size=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
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

def predict_result(model, scaler, home_team_id: str, away_team_id: str, odds: tuple[float, float, float]) -> tuple[str, float]:
    """Predict match result using the PyTorch model."""
    model.eval()
    with torch.no_grad():
        # Prepare input data
        data = pd.DataFrame({
            'home_team_id': [home_team_id],
            'away_team_id': [away_team_id],
            'home_odds': [odds[0]],
            'draw_odds': [odds[1]],
            'away_odds': [odds[2]]
        })
        
        data['home_team_encoded'] = data['home_team_id'].astype('category').cat.codes
        data['away_team_encoded'] = data['away_team_id'].astype('category').cat.codes

        X = data[['home_team_encoded', 'away_team_encoded', 'home_odds', 'draw_odds', 'away_odds']]
        X_scaled = scaler.transform(X)
        tensor = torch.FloatTensor(X_scaled)
        probabilities = model(tensor)
        
        # Get prediction and probability
        prediction = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0][prediction].item()
        
        # Convert prediction to result string
        result_map = {0: 'loss', 1: 'draw', 2: 'win'}
        result = result_map[prediction]

    return result, probability

if __name__ == "__main__":
    # Read CSV file, skipping the first row which appears to be corrupted
    df = pd.read_csv('processed_data.csv', header=None, skiprows=1)

    # Assign the column names to the dataframe
    column_names = [
        'home_team_id', 'away_team_id', 'year', 'month', 'day', 'hour',
        'home_odds', 'home_odds_min', 'home_odds_max',
        'draw_odds', 'draw_odds_min', 'draw_odds_max',
        'away_odds', 'away_odds_min', 'away_odds_max',
        'home_score', 'away_score', 'result'
    ]
    df.columns = column_names

    # Display basic information about the dataset
    print("Dataset Info:")
    print(df.info())

    # Show the first few rows
    print("\nFirst few rows:")
    print(df.head())

    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    model, scaler = train_model(df)
