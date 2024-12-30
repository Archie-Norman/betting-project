import pandas as pd

# Load the data from a CSV file
df = pd.read_csv("<prediction output>")

# Drop rows with any NaN values
df = df.dropna()

# Initial balance
initial_balance = 42.24

# Minimum Kelly fraction threshold
kelly_threshold = 0.03  # Bets with a fraction below this will be excluded

# Function to calculate the Kelly Criterion bet size and fractions
def kelly_criterion(prob_win, prob_draw, prob_loss, win_odds, draw_odds, loss_odds, initial_balance, num_bets_per_day, max_exposure):
    # Calculate Kelly fraction for each outcome (win, draw, loss)
    f_win = (win_odds * (prob_win / 100) - (1 - prob_win / 100)) / win_odds
    f_draw = (draw_odds * (prob_draw / 100) - (1 - prob_draw / 100)) / draw_odds
    f_loss = (loss_odds * (prob_loss / 100) - (1 - prob_loss / 100)) / loss_odds
    
    # Normalize the bets (must be between 0 and 1)
    f_win = max(0, min(f_win, 1))
    f_draw = max(0, min(f_draw, 1))
    f_loss = max(0, min(f_loss, 1))
    
    # Adjust Kelly fraction by the number of bets for the day
    adjusted_f_win = f_win / num_bets_per_day
    adjusted_f_draw = f_draw / num_bets_per_day
    adjusted_f_loss = f_loss / num_bets_per_day

    # Calculate the maximum exposure for the day (25% of initial balance)
    max_daily_bet = initial_balance * max_exposure
    
    # Determine the best bet
    bet_type = ''
    bet_amount = 0
    kelly_fraction = 0  # Variable to hold the Kelly fraction for the bet

    # Select the best bet and calculate the bet amount
    if adjusted_f_win >= adjusted_f_draw and adjusted_f_win >= adjusted_f_loss:
        bet_type = 'Win'
        kelly_fraction = adjusted_f_win
    elif adjusted_f_draw >= adjusted_f_win and adjusted_f_draw >= adjusted_f_loss:
        bet_type = 'Draw'
        kelly_fraction = adjusted_f_draw
    else:
        bet_type = 'Loss'
        kelly_fraction = adjusted_f_loss
    
    # Exclude bets with a Kelly fraction below the threshold
    if kelly_fraction < kelly_threshold:
        bet_type = 'None'  # No bet placed
        bet_amount = 0
    else:
        bet_amount = initial_balance * kelly_fraction
        # Ensure the bet amount doesn't exceed the max exposure for the day
        bet_amount = min(bet_amount, max_daily_bet)
        # Round the bet amount to two decimal places
        bet_amount = round(bet_amount, 2)
    
    return bet_type, bet_amount, kelly_fraction

# Group the DataFrame by date to count the number of bets for each day
df['date'] = pd.to_datetime(df['timestamp']).dt.date
bets_per_day = df.groupby('date').size()

# Maximum exposure limit (25% of the balance per day)
max_exposure = 0.25

# Apply the function to the DataFrame and store Kelly fraction
df['best_bet'], df['bet_amount'], df['kelly_fraction'] = zip(*df.apply(lambda row: kelly_criterion(
    row['prob_win'], row['prob_draw'], row['prob_loss'],
    row['win_odds'], row['draw_odds'], row['loss_odds'], 
    initial_balance, bets_per_day[row['date']], max_exposure), axis=1))

# Filter out rows with no bet placed
df_filtered = df[df['best_bet'] != 'None']

# Display the results
print(df_filtered[['date', 'team_home', 'team_away', 'best_bet', 'bet_amount', 'kelly_fraction']])

# Calculate and print the total winnings
total_winnings = df_filtered['bet_amount'].sum()
print("Total bet_amount:", total_winnings)

# Save the updated DataFrame to a CSV file
df_filtered.to_csv("stake_ammount.csv", index=False)
