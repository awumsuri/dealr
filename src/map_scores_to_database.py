from pymongo import MongoClient
from populate.match_scores import MatchScores

def update_match_scores():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:7777/')
    db = client['dealr']
    events_collection = db['events']
    
    # Initialize MatchScores
    match_scores = MatchScores()
    
    # Get all events from database
    events = events_collection.find({})
    
    for event in events:
        # Extract match details from event and ensure they're strings
        home_team = str(event['home_team'])
        away_team = str(event['away_team'])
        commence_time = str(event['commence_time'])
        
        # Get match result using existing MatchScores class
        match_data = match_scores.get_match_result(home_team, away_team, commence_time)
        
        if (match_data is None):
            print(f"No match data found for {home_team} vs {away_team} on {commence_time}")
            # Save missing match to events_missing collection
            missing_match = {
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time
            }
            db['events_missing'].insert_one(missing_match)
            continue
        
        print(match_data)
        if match_data:
            # Prepare update data
            update_data = {
                '$set': {
                    'WLD': match_data['result'],
                    'home_goals': match_data['home_goals'],
                    'away_goals': match_data['away_goals']
                }
            }
            
            # Update the document
            events_collection.update_one(
                {'_id': event['_id']},
                update_data
            )

if __name__ == "__main__":
    update_match_scores()
