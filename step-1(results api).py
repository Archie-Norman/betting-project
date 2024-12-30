import requests
import pandas as pd
import time

# Define your RapidAPI key and headers
api_key = '<api key>'  # Replace with your RapidAPI key
comp_ids = range(1, 161)  # Competition IDs from 1 to 160
headers = {
    'X-RapidAPI-Key': api_key,
    'X-RapidAPI-Host': 'football-web-pages1.p.rapidapi.com'
}

# Define the base URL for the Fixtures/Results endpoint
base_url = 'https://football-web-pages1.p.rapidapi.com/fixtures-results.json'

# Initialize an empty DataFrame to store results
all_results = pd.DataFrame()

# Loop through each competition ID
for comp_id in comp_ids:
    # Define parameters for the current competition ID
    params = {
        'comp': comp_id  # Competition ID
    }

    while True:  # Loop to handle retry on 429 status
        # Make the API request
        response = requests.get(base_url, headers=headers, params=params)

        # Check if we need to retry due to rate limiting
        if response.status_code == 429:
            print("Rate limit reached. Waiting for 60 seconds before retrying...")
            time.sleep(70)  # Wait 60 seconds and retry
            continue

        # Break the loop if the request was successful or an error other than 429 occurred
        if response.status_code == 200:
            data = response.json()
            break
        else:
            print(f"Failed to fetch fixtures/results for competition ID {comp_id}. Status code: {response.status_code}")
            print("Response Content:", response.text)
            break

    # Proceed if we have data from a successful request
    fixtures_results = data.get('fixtures-results', {}).get('matches', [])
    
    if fixtures_results:
        # Convert the data into a DataFrame
        df = pd.DataFrame(fixtures_results)
        
        # Normalize the nested dictionaries for home and away teams
        home_teams = df['home-team'].apply(pd.Series)
        away_teams = df['away-team'].apply(pd.Series)

        # Select the columns, checking if 'venue' is available
        columns_to_include = ['date', 'time']
        if 'venue' in df.columns:
            columns_to_include.append('venue')
        
        # Combine the data into a single DataFrame
        results = pd.concat([
            df[columns_to_include], 
            home_teams[['name', 'score']].rename(columns={'name': 'home_team', 'score': 'home_score'}), 
            away_teams[['name', 'score']].rename(columns={'name': 'away_team', 'score': 'away_score'})
        ], axis=1)
        results['competition_id'] = comp_id  # Add the competition_id column
        
        # Append results to the all_results DataFrame
        all_results = pd.concat([all_results, results], ignore_index=True)
        
        print(f"Fetched fixtures/results for competition ID {comp_id}")
    else:
        print(f"No fixtures found for competition ID {comp_id}.")

# Display the combined DataFrame with all results
print(all_results)

# Save the results to a CSV file
all_results.to_csv("fullres.csv", index=False)
