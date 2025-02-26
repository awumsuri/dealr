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
    """Train a PyTorch neural network model with WeightedCrossEntropyLoss using odds."""
    # Select features (everything except the target 'result')
    feature_cols = [
        'home_team_id', 'away_team_id', 'year', 'month', 'day', 'hour',
        'home_odds_mean', 'home_odds_min', 'home_odds_max',
        'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
        'away_odds_mean', 'away_odds_min', 'away_odds_max',
        'expected_winner'
    ]
    X = df[feature_cols]
    y = df['result']  # 0=Win,1=Draw,2=Loss

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Validation/Test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train.values)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val.values)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test.values)

    # Initialize model, custom loss, optimizer, scheduler
    model = BettingNN(input_size=X_train.shape[1])
    criterion = WeightedCrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    num_epochs = 100
    batch_size = 32
    n_batches = len(X_train) // batch_size

    # Indices for columns that hold the mean odds for home/draw/away
    # Adjust these to match your actual layout
    HOME_ODDS_MEAN_COL = 6   # 'home_odds_mean'
    DRAW_ODDS_MEAN_COL = 9   # 'draw_odds_mean'
    AWAY_ODDS_MEAN_COL = 12  # 'away_odds_mean'

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_X = X_train[start_idx:end_idx]  # shape [B, input_size]
            batch_y = y_train[start_idx:end_idx]  # shape [B]

            # Slice the relevant odds columns for weighting: shape [B, 3]
            batch_odds = batch_X[:, [HOME_ODDS_MEAN_COL, DRAW_ODDS_MEAN_COL, AWAY_ODDS_MEAN_COL]]

            optimizer.zero_grad()
            outputs = model(batch_X)  # shape [B, 3]
            loss = criterion(outputs, batch_y, batch_odds)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            # Slice odds for validation
            val_odds = X_val[:, [HOME_ODDS_MEAN_COL, DRAW_ODDS_MEAN_COL, AWAY_ODDS_MEAN_COL]]
            val_loss = criterion(val_outputs, y_val, val_odds)

            # Compute accuracy
            val_preds = torch.argmax(val_outputs, dim=1)
            val_correct = (val_preds == y_val).sum().item()
            val_accuracy = val_correct / len(y_val)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % 10 == 0:
            train_accuracy = correct_predictions / total_samples
            print(f"[Epoch {epoch + 1}/{num_epochs}] "
                  f"Train Loss: {total_loss/n_batches:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")

    # Load best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    return model, scaler, (X_test, y_test)

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

def evaluate_model(model, X_test, y_test):
    """
    Evaluate on test set (unweighted for demonstration).
    If you want the weighted loss, you can slice out the test odds
    and call WeightedCrossEntropyLoss with them.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean().item()
    return accuracy

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

    model, scaler, (X_test, y_test)  = train_model(df)
