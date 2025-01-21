import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from process_engineering import process_row_features
from prediction_data import raw_prediction_data


class BettingNN(nn.Module):
    def __init__(self, input_size):
        super(BettingNN, self).__init__()
        self.layers = nn.Sequential(
            # Wider initial layers for better feature learning
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),  # LeakyReLU often performs better than ReLU
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 3),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, outputs, targets, odds_batch):
        """
        :param outputs: model logits, shape [batch_size, num_classes]
        :param targets: ground-truth labels, shape [batch_size]
        :param odds_batch: odds for the outcomes of *this batch*, shape [batch_size, num_classes]
        """
        # Convert model outputs to log probabilities
        log_probs = torch.log_softmax(outputs, dim=1)

        # Gather the log-probability for the correct class
        loss = -log_probs.gather(1, targets.unsqueeze(1))  # shape [batch_size, 1]

        # Gather odds for the correct class
        odds_for_true_class = odds_batch.gather(1, targets.unsqueeze(1))  # shape [batch_size, 1]

        # Multiply element-wise: more weighting for higher odds
        weighted_loss = loss * odds_for_true_class

        return weighted_loss.mean()        
   

def train_model(df: pd.DataFrame):
    """Train a PyTorch neural network model with WeightedCrossEntropyLoss using odds."""
    # Select features (everything except the target 'result')
    feature_cols = [
        'home_team_id', 'away_team_id', 'year', 'month', 'day', 'hour',
        'home_odds_mean', 'home_odds_min', 'home_odds_max',
        'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
        'away_odds_mean', 'away_odds_min', 'away_odds_max'
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
    patience = 100
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

def predict_result(model, scaler, single_df: pd.DataFrame):
    """
    Predict match result using the trained model.
    single_df: DataFrame with same columns as training except 'result'
    Returns (prediction_str, probability)
    """
    model.eval()
    with torch.no_grad():
        # Scale input
        X_scaled = scaler.transform(single_df)
        tensor = torch.FloatTensor(X_scaled)
        logits = model(tensor)  # shape [1, 3] for a single row
        probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        pred_class = torch.argmax(probs, dim=1).item()
        prob_value = probs[0, pred_class].item()

    # Map int->class
    # 0: Win (Home), 1: Draw, 2: Loss (Home)
    result_map = {0: 'Win', 1: 'Draw', 2: 'Loss'}
    return result_map[pred_class], prob_value


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
    # Example usage (you'll need to provide your own DataFrame with historical data)
    # historical_df = your_data_loading_function()
    # model, scaler = train_model(historical_df)
    # result, probability = predict_result(model, scaler, "Team A", "Team B")
    # print(f"Predicted result: {result} with probability: {probability:.2f}")
   
    # Read CSV file, skipping the first row which appears to be corrupted
    df = pd.read_csv('processed_data.csv')

    # # Assign the column names to the dataframe
    # column_names = [
    #     'home_team_id', 'away_team_id', 'year', 'month', 'day', 'hour',
    #     'home_odds', 'home_odds_min', 'home_odds_max',
    #     'draw_odds', 'draw_odds_min', 'draw_odds_max',
    #     'away_odds', 'away_odds_min', 'away_odds_max',
    #     'expected_winner', 'result'
    # ]
    # df.columns = column_names

    # Display basic information about the dataset
    print("Dataset Info:")
    print(df.info())

    # Show the first few rows
    print("\nFirst few rows:")
    print(df.head())

    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())   

    model, scaler, (X_test, y_test) = train_model(df)
    
     # Evaluate final test accuracy
    test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")    
  
    Xp = process_row_features(raw_prediction_data)
    
    # Define the column names (excluding result and scores)
    feature_columns = [
        'home_team_id', 'away_team_id', 'year', 'month', 'day', 'hour',
        'home_odds_mean', 'home_odds_min', 'home_odds_max',
        'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
        'away_odds_mean', 'away_odds_min', 'away_odds_max'
       
    ]
    
    # Create DataFrame with proper column names
    data = pd.DataFrame([Xp], columns=feature_columns)
    
  
    
    resultP = predict_result(model, scaler, data)
    print(resultP)
