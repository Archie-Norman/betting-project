import pandas as pd
import numpy as np
import requests


results = pd.read_csv("<output from the api>")

# List of competition IDs to drop
competition_ids_to_drop = [
    21, 22, 23,24, 25, 116, 28, 32, 27, 33, 26, 30, 36, 
    35, 34, 81, 156,106,107
]

# Drop rows where the 'competition_id' is in the list
results = results[~results['competition_id'].isin(competition_ids_to_drop)]



results = results.drop_duplicates()

# Convert date and time to a single timestamp column
results['timestamp'] = pd.to_datetime(results['date'] + ' ' + results['time'], errors='coerce')

# Remove rows where the date is after today's date
results = results[results['timestamp'] <= pd.Timestamp.now() + pd.Timedelta(days=3)]


# Drop rows where `competition_id` is 87, 154, or 86
results = results[~results['competition_id'].isin([87, 154, 86])]

# Drop duplicates based on specific columns
results = results.drop_duplicates(subset=['timestamp', 'home_team', 'competition_id'])

# Transform the DataFrame
rows = []

for index, row in results.iterrows():
    # Append home team data
    rows.append({
        'date': row['date'],
        'time': row['time'],
        'venue': row['venue'],
        'team': row['home_team'],
        'against': row['away_team'],
        'away_or_home': 1,
        'goals': row['home_score'],
        'conceded': row['away_score'],
        'competition_id': row['competition_id']
    })
    
    # Append away team data
    rows.append({
        'date': row['date'],
        'time': row['time'],
        'venue': row['venue'],
        'team': row['away_team'],
        'against': row['home_team'],
        'away_or_home': 0,
        'goals': row['away_score'],
        'conceded': row['home_score'],
        'competition_id': row['competition_id']
    })



# Create a new DataFrame with the transformed data
transformed_df = pd.DataFrame(rows)

# Convert date and time to a single timestamp column
transformed_df['timestamp'] = pd.to_datetime(transformed_df['date'] + ' ' + transformed_df['time'])
transformed_df = transformed_df.sort_values(by=['team', 'timestamp'])


# Rearranging the columns to match the desired order
transformed_df = transformed_df[['timestamp', 'venue', 'team', 'against', 'away_or_home', 'goals', 'conceded','competition_id']]

# Determine the win/loss/draw status
def determine_result(row):
    if row['goals'] > row['conceded']:
        return 2  # Win
    elif row['goals'] < row['conceded']:
        return 0  # Loss
    else:
        return 1  # Draw

# Apply the function to create a new column 'win_or_loss'
transformed_df['win_or_loss'] = transformed_df.apply(determine_result, axis=1)

# Rearranging the columns to include 'win_or_loss'
transformed_df = transformed_df[['timestamp', 'venue', 'team', 'against', 'away_or_home', 'goals', 'conceded', 'win_or_loss','competition_id']]


# Initialize the 'total_conceded' column with 0
transformed_df['conceded_before_game'] = 0

# Sort the data by team and timestamp to ensure cumulative calculation order
transformed_df = transformed_df.sort_values(by=['team', 'timestamp'])

# Calculate the cumulative total of conceded goals for each team
# Shift the result by 1 to exclude the current row's 'conceded' from the cumulative total
transformed_df['conceded_before_game'] = transformed_df.groupby('team')['conceded'].cumsum().shift(1).fillna(0).astype(int)

# Initialize the 'total_conceded' column with 0
transformed_df['goals_before_game'] = 0

# Sort the data by team and timestamp to ensure cumulative calculation order
transformed_df = transformed_df.sort_values(by=['team', 'timestamp'])

# Calculate the cumulative total of conceded goals for each team
# Shift the result by 1 to exclude the current row's 'conceded' from the cumulative total
transformed_df['goals_before_game'] = transformed_df.groupby('team')['goals'].cumsum().shift(1).fillna(0).astype(int)

# Initialize a points dictionary to accumulate points for each team
points_dict = {team: 0 for team in transformed_df['team'].unique()}

# Function to determine result and points
def determine_result_and_points(row):
    if row['goals'] > row['conceded']:
        return 2, 3  # Win: Result = 2, Points = 3
    elif row['goals'] < row['conceded']:
        return 0, 0  # Loss: Result = 0, Points = 0
    else:
        return 1, 1  # Draw: Result = 1, Points = 1

# List to hold results for win/loss and points
results = []

# Calculate results and update points for each match
for index, row in transformed_df.iterrows():
    # Determine result and points for the current match
    result, points = determine_result_and_points(row)
    
    # Update points for the team
    points_dict[row['team']] += points
    
    # Append the current result and cumulative points for the team
    results.append((result, points_dict[row['team']]))

# Add the results as new columns in the DataFrame
transformed_df['win_or_loss'] = [determine_result_and_points(row)[0] for _, row in transformed_df.iterrows()]
transformed_df['points'] = [points_dict[row['team']] for _, row in transformed_df.iterrows()]

# Rearranging the columns to include 'win_or_loss' and 'points'
transformed_df = transformed_df[['timestamp', 'venue', 'team', 'against', 'away_or_home', 'goals', 'conceded', 'win_or_loss', 'points','conceded_before_game','goals_before_game','competition_id']]


# Initialize the 'total_conceded' column with 0
transformed_df['points_before_game'] = 0

# Sort the data by team and timestamp to ensure cumulative calculation order
transformed_df = transformed_df.sort_values(by=['team', 'timestamp'])

# Calculate the cumulative total of conceded goals for each team
# Shift the result by 1 to exclude the current row's 'conceded' from the cumulative total
transformed_df['points_before_game'] = transformed_df.groupby('team')['points'].cumsum().shift(1).fillna(0).astype(int)


# Initialize Elo ratings for each team
initial_elo = 1500  # Starting Elo rating
elo_ratings = {team: initial_elo for team in transformed_df['team'].unique()}

# Elo rating update parameters
K = 30  # K-factor

# Function to update Elo ratings
def update_elo(team1, team2, score1, score2):
    # Calculate expected scores
    expected1 = 1 / (1 + 10 ** ((elo_ratings[team2] - elo_ratings[team1]) / 400))
    expected2 = 1 / (1 + 10 ** ((elo_ratings[team1] - elo_ratings[team2]) / 400))

    # Determine actual scores
    if score1 > score2:
        actual1, actual2 = 1, 0  # Team1 wins
    elif score1 < score2:
        actual1, actual2 = 0, 1  # Team2 wins
    else:
        actual1, actual2 = 0.5, 0.5  # Draw

    # Update ratings
    elo_ratings[team1] += K * (actual1 - expected1)
    elo_ratings[team2] += K * (actual2 - expected2)

# Create a new column for Elo ratings in transformed_df
transformed_df['elo_rating'] = 0.0  # Initialize with zero or any default value

# Iterate through each game and update Elo ratings
for index, row in transformed_df.iterrows():
    if row['away_or_home'] == 1:  # Home team
        home_team = row['team']
        away_team = transformed_df.loc[index + 1, 'team']  # Next row should be the away team
        home_score = row['goals']
        away_score = transformed_df.loc[index + 1, 'goals']
        
        # Update Elo ratings
        update_elo(home_team, away_team, home_score, away_score)

        # Assign the updated Elo ratings to the DataFrame
        transformed_df.at[index, 'elo_rating'] = elo_ratings[home_team]
        transformed_df.at[index + 1, 'elo_rating'] = elo_ratings[away_team]

# Rearranging the columns to include 'elo_rating'
transformed_df = transformed_df[['timestamp', 'venue', 'team', 'against', 'away_or_home', 'goals', 'conceded', 'win_or_loss', 'points', 'elo_rating', 'points_before_game' ,'conceded_before_game','goals_before_game','competition_id']]

# Filter for home games only
home_games = transformed_df[transformed_df['away_or_home'] == 1].copy()

home_games['games_played'] = 0  # Start with a column initialized to 0

# Count total home games for each team using cumsum
home_games.loc[home_games['away_or_home'] == 1, 'games_played'] = 1  # Set home games to 1
home_games['games_played'] = home_games.groupby('team')['games_played'].cumsum()  # Cumulative sum for home games

# Initialize a points dictionary to accumulate points for each team
points_dict = {team: 0 for team in home_games['team'].unique()}

# Function to determine result and points
def determine_result_and_points(row):
    if row['goals'] > row['conceded']:
        return 2, 3  # Win: Result = 2, Points = 3
    elif row['goals'] < row['conceded']:
        return 0, 0  # Loss: Result = 0, Points = 0
    else:
        return 1, 1  # Draw: Result = 1, Points = 1

# Initialize lists to hold results and cumulative points
results = []
cumulative_points = []

# Calculate results and update points for each match
for index, row in home_games.iterrows():
    # Determine result and points for the current match
    result, points = determine_result_and_points(row)
    
    # Update points for the team
    points_dict[row['team']] += points
    
    # Append the current result and cumulative points for the team
    results.append(result)
    cumulative_points.append(points_dict[row['team']])

# Add the results and cumulative points as new columns in the DataFrame
home_games['win_or_loss_home'] = results
home_games['points_home'] = cumulative_points

home_games['games_to_points_ratio_home'] = home_games['games_played'] / home_games['points_home'].replace(0, np.nan)  # or replace(0, 1)
home_games['games_to_points_ratio_home'].fillna(0)

# Step 2: Merge the DataFrames on timestamp and team
transformed_df = transformed_df.merge(
    home_games[['timestamp', 'team', 'games_to_points_ratio_home']], 
    on=['timestamp', 'team'], 
    how='left'  # Use 'left' to keep all entries from transformed_df
)



# Step 1: Sort the DataFrame by team and timestamp
transformed_df.sort_values(by=['team', 'timestamp'], inplace=True)

# Step 2: Forward fill the NaN values within each team
transformed_df['games_to_points_ratio_home'] = transformed_df.groupby('team')['games_to_points_ratio_home'].ffill()

# Step 3: Fill any leading NaNs (if a team has no previous entries) with 0
transformed_df['games_to_points_ratio_home'].fillna(0, inplace=True)



# Filter for away games only
away_games = transformed_df[transformed_df['away_or_home'] == 0].copy()

away_games['games_played'] = 0  # Start with a column initialized to 0

# Count total away games for each team using cumsum
away_games.loc[away_games['away_or_home'] == 0, 'games_played'] = 1  # Set home games to 1
away_games['games_played'] = away_games.groupby('team')['games_played'].cumsum()  # Cumulative sum for home games

# Initialize a points dictionary to accumulate points for each team
points_dict = {team: 0 for team in away_games['team'].unique()}

# Function to determine result and points
def determine_result_and_points(row):
    if row['goals'] > row['conceded']:
        return 2, 3  # Win: Result = 2, Points = 3
    elif row['goals'] < row['conceded']:
        return 0, 0  # Loss: Result = 0, Points = 0
    else:
        return 1, 1  # Draw: Result = 1, Points = 1

# Initialize lists to hold results and cumulative points
results = []
cumulative_points = []

# Calculate results and update points for each match
for index, row in away_games.iterrows():
    # Determine result and points for the current match
    result, points = determine_result_and_points(row)
    
    # Update points for the team
    points_dict[row['team']] += points
    
    # Append the current result and cumulative points for the team
    results.append(result)
    cumulative_points.append(points_dict[row['team']])

# Add the results and cumulative points as new columns in the DataFrame
away_games['win_or_loss_home'] = results
away_games['points_home'] = cumulative_points

away_games['games_to_points_ratio_away'] = away_games['games_played'] / away_games['points_home'].replace(0, np.nan)  # or replace(0, 1)
away_games['games_to_points_ratio_away'].fillna(0, inplace=True)

# Step 2: Merge the DataFrames on timestamp and team
transformed_df = transformed_df.merge(
    away_games[['timestamp', 'team', 'games_to_points_ratio_away']], 
    on=['timestamp', 'team'], 
    how='left'  # Use 'left' to keep all entries from transformed_df
)



# Step 1: Sort the DataFrame by team and timestamp
transformed_df.sort_values(by=['team', 'timestamp'], inplace=True)

# Step 2: Forward fill the NaN values within each team
transformed_df['games_to_points_ratio_away'] = transformed_df.groupby('team')['games_to_points_ratio_away'].ffill()

# Step 3: Fill any leading NaNs (if a team has no previous entries) with 0
transformed_df['games_to_points_ratio_away'].fillna(0, inplace=True)


# Step 1: Sort the DataFrame by team and timestamp
transformed_df.sort_values(by=['team', 'timestamp'], inplace=True)

# Step 2: Group by team and forward fill the elo_rating
transformed_df['last_games_elo'] = transformed_df.groupby('team')['elo_rating'].shift()
transformed_df['last_games_elo'].fillna(1500, inplace=True)

transformed_df.sort_values(by=['timestamp', 'venue','away_or_home'], inplace=True)

print(transformed_df)












# Step 3: Create against_last_elo column based on away_or_home
def get_against_last_elo(row):
    # If it's an away game (0), get the elo from the row below
    if row['away_or_home'] == 0:
        return transformed_df['last_games_elo'].shift(-1)[row.name]
    # If it's a home game (1), get the elo from the row above
    else:
        return transformed_df['last_games_elo'].shift(1)[row.name]

transformed_df['against_last_elo'] = transformed_df.apply(get_against_last_elo, axis=1)

transformed_df.sort_values(by=['timestamp', 'venue','away_or_home'], inplace=True)


transformed_df['elo_diff'] = transformed_df['last_games_elo'] - transformed_df['against_last_elo']


# oppo games_to_points_ratio_home, oppo games_to_points_ratio_away

# Step 3: Create against_last_elo column based on away_or_home
def get_against_last_elo(row):
    # If it's an away game (0), get the elo from the row below
    if row['away_or_home'] == 0:
        return transformed_df['points_before_game'].shift(-1)[row.name]
    # If it's a home game (1), get the elo from the row above
    else:
        return transformed_df['points_before_game'].shift(1)[row.name]

transformed_df['against_points_before_game'] = transformed_df.apply(get_against_last_elo, axis=1)

def get_against_last_elo(row):
    # If it's an away game (0), get the elo from the row below
    if row['away_or_home'] == 0:
        return transformed_df['conceded_before_game'].shift(-1)[row.name]
    # If it's a home game (1), get the elo from the row above
    else:
        return transformed_df['conceded_before_game'].shift(1)[row.name]

transformed_df['against_conceded_before_game'] = transformed_df.apply(get_against_last_elo, axis=1)

def get_against_last_elo(row):
    # If it's an away game (0), get the elo from the row below
    if row['away_or_home'] == 0:
        return transformed_df['goals_before_game'].shift(-1)[row.name]
    # If it's a home game (1), get the elo from the row above
    else:
        return transformed_df['goals_before_game'].shift(1)[row.name]

transformed_df['against_goals_before_game'] = transformed_df.apply(get_against_last_elo, axis=1)

####
def get_against_last_elo(row):
    # If it's an away game (0), get the elo from the row below
    if row['away_or_home'] == 0:
        return transformed_df['games_to_points_ratio_home'].shift(-1)[row.name]
    # If it's a home game (1), get the elo from the row above
    else:
        return transformed_df['games_to_points_ratio_home'].shift(1)[row.name]

transformed_df['against_games_to_points_ratio_home'] = transformed_df.apply(get_against_last_elo, axis=1)

def get_against_last_elo(row):
    # If it's an away game (0), get the elo from the row below
    if row['away_or_home'] == 0:
        return transformed_df['games_to_points_ratio_away'].shift(-1)[row.name]
    # If it's a home game (1), get the elo from the row above
    else:
        return transformed_df['games_to_points_ratio_away'].shift(1)[row.name]

transformed_df['against_games_to_points_ratio_away'] = transformed_df.apply(get_against_last_elo, axis=1)




print(transformed_df)
# Print column names as a list
print(list(transformed_df.columns))

#transformed_df.to_csv("trans.csv", index=False)



# Split into home and away DataFrames based on `away_or_home` values
home_df = transformed_df[transformed_df['away_or_home'] == 1].drop(columns=['away_or_home']).add_suffix('_home')
away_df = transformed_df[transformed_df['away_or_home'] == 0].drop(columns=['away_or_home']).add_suffix('_away')

# Merge on timestamp and ensure team and opponent match up correctly
combined_df = pd.merge(
    home_df,
    away_df,
    left_on=['timestamp_home', 'team_home', 'against_home'],
    right_on=['timestamp_away', 'against_away', 'team_away'],
    suffixes=('_home', '_away')
)

# Drop duplicated timestamp column and rename if desired
transformed_df = combined_df.drop(columns=['timestamp_away']).rename(columns={'timestamp_home': 'timestamp'})

transformed_df['total_goals'] = transformed_df['goals_home'] + transformed_df['conceded_home']


print(transformed_df)







# Ensure the 'team_home' and 'team_away' columns are treated as strings
transformed_df['team_home'] = transformed_df['team_home'].astype(str)
transformed_df['against_home'] = transformed_df['against_home'].astype(str)

# Sort the data by timestamp to ensure chronological order
transformed_df['timestamp'] = pd.to_datetime(transformed_df['timestamp'])
transformed_df = transformed_df.sort_values('timestamp')

# Initialize a dictionary to track total games played by each team
total_games = {}

# Function to count games
def count_total_games(team, total_games_dict):
    if team not in total_games_dict:
        total_games_dict[team] = 0
    total_games_dict[team] += 1
    return total_games_dict[team]

# Create a function to apply counting logic for total games
def compute_total_games(row, total_games_dict):
    # Count for home team
    row['games_played_home_total'] = count_total_games(row['team_home'], total_games_dict)
    # Count for away team
    row['games_played_away_total'] = count_total_games(row['against_home'], total_games_dict)
    return row

# Apply the counting function row-wise
transformed_df = transformed_df.apply(lambda row: compute_total_games(row, total_games), axis=1)

# Save or view the updated DataFrame
print(transformed_df[['timestamp', 'team_home', 'games_played_home_total', 'against_home', 'games_played_away_total']])








# Save the combined DataFrame to CSV
transformed_df.to_csv("trans.csv", index=False)

