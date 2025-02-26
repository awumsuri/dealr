import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from process_engineering import process_row_features
from prediction_data import raw_prediction_data

# Custom loss combining focal loss, class weights, and odds weighting
class FocalWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalWeightedCrossEntropyLoss, self).__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma

    def forward(self, outputs, targets, odds_batch):
        log_probs = torch.log_softmax(outputs, dim=1)
        ce_loss = -log_probs.gather(1, targets.unsqueeze(1))  # [B, 1]
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply odds weighting
        odds_weight = odds_batch.gather(1, targets.unsqueeze(1))
        weighted_loss = focal_loss * odds_weight
        
        # Apply class weights
        if self.alpha is not None:
            alpha_weight = self.alpha[targets].unsqueeze(1)
            weighted_loss *= alpha_weight
            
        return weighted_loss.mean()

class BettingNN(nn.Module):
    def __init__(self, num_teams, numerical_input_size, embedding_dim=10):
        super(BettingNN, self).__init__()
        self.team_embed = nn.Embedding(num_teams, embedding_dim)
        self.numerical_input_size = numerical_input_size
        
        self.layers = nn.Sequential(
            nn.Linear(2*embedding_dim + numerical_input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(32, 3)
        )

    def forward(self, x_teams, x_numerical):
        home_emb = self.team_embed(x_teams[:, 0])
        away_emb = self.team_embed(x_teams[:, 1])
        combined = torch.cat([home_emb, away_emb, x_numerical], dim=1)
        return self.layers(combined)

def train_model(df: pd.DataFrame):
    # Add team ID normalization at the start
    # Remap team IDs to ensure they start from 0 and are continuous
    team_ids = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
    team_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(team_ids))}
    
    # Create copies to avoid modifying original data
    df = df.copy()
    df['home_team_id'] = df['home_team_id'].map(team_id_map)
    df['away_team_id'] = df['away_team_id'].map(team_id_map)

    # Feature selection
    team_cols = ['home_team_id', 'away_team_id']
    numerical_cols = [
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
        'home_prob', 'draw_prob', 'away_prob',
        'home_odds_mean', 'home_odds_min', 'home_odds_max',
        'draw_odds_mean', 'draw_odds_min', 'draw_odds_max',
        'away_odds_mean', 'away_odds_min', 'away_odds_max'
    ]
    
    # Split data
    X_teams = df[team_cols]
    X_numerical = df[numerical_cols]
    y = df['result'].values
    
    # Chronological split
    split_idx = int(0.8 * len(df))
    X_teams_train, X_teams_test = X_teams.iloc[:split_idx], X_teams.iloc[split_idx:]
    X_numerical_train, X_numerical_test = X_numerical.iloc[:split_idx], X_numerical.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_numerical_train = scaler.fit_transform(X_numerical_train)
    X_numerical_test = scaler.transform(X_numerical_test)
    
    # Further split for validation
    X_teams_train, X_teams_val, X_numerical_train, X_numerical_val, y_train, y_val = train_test_split(
        X_teams_train, X_numerical_train, y_train, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    def to_tensor(data, dtype=torch.float):
        return torch.tensor(data.values if isinstance(data, pd.DataFrame) else data, dtype=dtype)
    
    X_teams_train = to_tensor(X_teams_train, torch.long)
    X_numerical_train = to_tensor(X_numerical_train)
    y_train = to_tensor(y_train, torch.long)
    
    X_teams_val = to_tensor(X_teams_val, torch.long)
    X_numerical_val = to_tensor(X_numerical_val)
    y_val = to_tensor(y_val, torch.long)
    
    X_teams_test = to_tensor(X_teams_test, torch.long)
    X_numerical_test = to_tensor(X_numerical_test)
    y_test = to_tensor(y_test, torch.long)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
    class_weights = torch.FloatTensor(class_weights)
    
    # Model setup
    num_teams = len(team_id_map)
    model = BettingNN(num_teams=num_teams, numerical_input_size=X_numerical_train.shape[1])
    criterion = FocalWeightedCrossEntropyLoss(alpha=class_weights, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    batch_size = 64
    
    for epoch in range(100):
        model.train()
        indices = torch.randperm(len(X_teams_train))  # Shuffle each epoch
        
        for i in range(0, len(X_teams_train), batch_size):
            idx = indices[i:i+batch_size]
            batch_teams = X_teams_train[idx]
            batch_numerical = X_numerical_train[idx]
            batch_y = y_train[idx]
            
            # Get odds columns (adjust indices based on numerical_cols)
            odds_indices = [numerical_cols.index('home_odds_mean'), 
                            numerical_cols.index('draw_odds_mean'), 
                            numerical_cols.index('away_odds_mean')]
            batch_odds = batch_numerical[:, odds_indices]
            
            optimizer.zero_grad()
            outputs = model(batch_teams, batch_numerical)
            loss = criterion(outputs, batch_y, batch_odds)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_odds = X_numerical_val[:, odds_indices]
            outputs = model(X_teams_val, X_numerical_val)
            val_loss = criterion(outputs, y_val, val_odds)
            val_acc = (outputs.argmax(1) == y_val).float().mean()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        test_odds = X_numerical_test[:, odds_indices]
        outputs = model(X_teams_test, X_numerical_test)
        test_acc = (outputs.argmax(1) == y_test).float().mean()
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return model, scaler, (X_teams_test, X_numerical_test, y_test)

def predict_result(model, scaler, single_df: pd.DataFrame):
    # Preprocess single_df similarly to training data
    # Ensure to apply cyclical encoding and probability calculations
    # Then scale numerical features and convert to tensors
    # Return prediction
    pass

if __name__ == "__main__":
    df = pd.read_csv('processed_data.csv')
    model, scaler, (X_test, X_numerical_test, y_test) = train_model(df)
    predict_result(model, scaler, X_test)