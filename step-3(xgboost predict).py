import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from datetime import datetime

# Load your dataset
data = pd.read_csv("<output from cleaning>")
data = data.drop_duplicates()

# Filter rows with sufficient games played
data = data[data['games_played_home_total'] > 5]
data = data[data['games_played_away_total'] > 5]

# Handle missing values
data['venue_home'] = data['venue_home'].fillna('Unknown')
data['venue_away'] = data['venue_away'].fillna('Unknown')

# Ensure timestamp column is in datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

# Split past and future data
future_data = data[data['timestamp'] > datetime.now()]
data = data[data['timestamp'] <= pd.Timestamp.now()]

# Apply a cut-off date
cut_off = pd.Timestamp("2024-09-11")
data = data[data['timestamp'] > cut_off]

# Select features and target variable
X = data[['venue_home', 
          'team_home', 
          'against_home', 
          'competition_id_home',
          'points_before_game_home', 
          'conceded_before_game_home', 
          'goals_before_game_home', 
          'games_to_points_ratio_home_home', 
          'games_to_points_ratio_away_home', 
          'last_games_elo_home', 
          'against_last_elo_home', 
          'elo_diff_home',
          'against_points_before_game_home',
          'against_conceded_before_game_home',
          'against_goals_before_game_home',
          'against_games_to_points_ratio_home_home',
          'against_games_to_points_ratio_away_home',
          'points_before_game_away', 
          'conceded_before_game_away', 
          'goals_before_game_away',  
          'last_games_elo_away', 
          'against_points_before_game_away',
          'against_conceded_before_game_away',
          'against_goals_before_game_away']]

y = data['win_or_loss_home']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home'], drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Train the base XGBoost model
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=5,
    reg_lambda=5,
    n_estimators=25,
    learning_rate=0.3
)

# Calibrate the model using isotonic regression
calibrated_model = CalibratedClassifierCV(estimator=xgb_model, method='isotonic', cv='prefit')

xgb_model.fit(X_train_scaled, y_train)
calibrated_model.fit(X_train_scaled, y_train)

# Predictions on test data
y_pred = calibrated_model.predict(X_test_scaled)
y_proba = calibrated_model.predict_proba(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Prepare future data
X_future = future_data[['venue_home', 
                        'team_home', 
                        'against_home', 
                        'competition_id_home',
                        'points_before_game_home', 
                        'conceded_before_game_home', 
                        'goals_before_game_home', 
                        'games_to_points_ratio_home_home', 
                        'games_to_points_ratio_away_home', 
                        'last_games_elo_home', 
                        'against_last_elo_home', 
                        'elo_diff_home',
                        'against_points_before_game_home',
                        'against_conceded_before_game_home',
                        'against_goals_before_game_home',
                        'against_games_to_points_ratio_home_home',
                        'against_games_to_points_ratio_away_home',
                        'points_before_game_away', 
                        'conceded_before_game_away', 
                        'goals_before_game_away',  
                        'last_games_elo_away', 
                        'against_points_before_game_away',
                        'against_conceded_before_game_away',
                        'against_goals_before_game_away']]

# One-hot encoding for future data
X_future_encoded = pd.get_dummies(X_future, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home'], drop_first=True)
X_future_encoded = X_future_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_future_scaled = scaler.transform(X_future_encoded)

# Make predictions on future data
future_predictions = calibrated_model.predict(X_future_scaled)
future_probabilities = calibrated_model.predict_proba(X_future_scaled)

# Map predictions and probabilities
result_mapping = {2: 'Win', 1: 'Draw', 0: 'Loss'}
future_data['predicted_win_or_loss_home'] = future_predictions
future_data['predicted_win_or_loss_home'] = future_data['predicted_win_or_loss_home'].map(result_mapping)
future_data['prob_win'] = (future_probabilities[:, 2] * 100).round(0).astype(int)
future_data['prob_draw'] = (future_probabilities[:, 1] * 100).round(0).astype(int)
future_data['prob_loss'] = (future_probabilities[:, 0] * 100).round(0).astype(int)

# Select only the desired columns
output_data = future_data[['timestamp', 'team_home', 'team_away', 'predicted_win_or_loss_home', 'prob_win', 'prob_draw', 'prob_loss']]

# Initialize the new columns with placeholder values (None or NaN)
output_data['win_odds'] = None   # You can replace None with a default value like NaN if needed
output_data['draw_odds'] = None  # Or any other default value
output_data['loss_odds'] = None  # Same here

# Save the updated future_data to a CSV file
output_data.to_csv("match_predictions.csv", index=False)

print("Predictions and probabilities added and saved to 'match_predictions.csv'")