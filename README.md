# betting-project

https://rapidapi.com/football-web-pages1-football-web-pages-default/api/football-web-pages1

the link above is to the API I used. it has a free tier which allows for 500 requests a day which is plenty. 

Step 1 (results API)

this is just a request to the API to get the results from every game in the season and upcoming games

Step 2 (cleaning)

filters out any cup games and adds loads of features which are below 

goals_home,conceded_home,win_or_loss_home,points_home,elo_rating_home,points_before_game_home,conceded_before_game_home,goals_before_game_home,competition_id_home,games_to_points_ratio_home_home,games_to_points_ratio_away_home,last_games_elo_home,against_last_elo_home,elo_diff_home,against_points_before_game_home,against_conceded_before_game_home,against_goals_before_game_home,against_games_to_points_ratio_home_home,against_games_to_points_ratio_away_home,venue_away,team_away,against_away,goals_away,conceded_away,win_or_loss_away,points_away,elo_rating_away,points_before_game_away,conceded_before_game_away,goals_before_game_away,competition_id_away,games_to_points_ratio_home_away,games_to_points_ratio_away_away,last_games_elo_away,against_last_elo_away,elo_diff_away,against_points_before_game_away,against_conceded_before_game_away,against_goals_before_game_away,against_games_to_points_ratio_home_away,against_games_to_points_ratio_away_away,total_goals,games_played_home_total,games_played_away_total

step 3 (xgboosting predict)

where the magic happens, just a simple xgboosting model using isotonic regression for the calibration but I will say one improvement would probably be adjusting for the class imbalance especially for draws.

this is also the longest step as you have to manully add the bookie's odds after

step 4 (EV bets)

the initial balance and scaling doesn't really work but I just ignore the amount it says to bet. this just tells me what has a positive expected outcome using the kelly criterion.
