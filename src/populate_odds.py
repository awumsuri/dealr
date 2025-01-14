from odds_api import OddsAPI
from pymongo import MongoClient
from datetime import datetime, timedelta
import time

class OddsPopulator:
    def __init__(self, mongodb_uri="mongodb://localhost:7777"):
        self.api = OddsAPI()
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['dealr']
        self.events_collection = self.db['events']

    def populate_historical_data(self, start_date_str="2020-07-26", sport_key="soccer_epl"):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        current_date = start_date

        while current_date < datetime.now():
            try:
                # Format date for API
                date_str = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Get events for the current date
                events_data = self.api.get_historical_events(
                    sport=sport_key,
                    date=date_str
                )

                # Process each event
                if 'data' in events_data:
                    for event in events_data['data']:
                        event_id = event['id']
                        date = event['commence_time']
                        
                        # Get historical odds for this event
                        odds_data = self.api.get_historical_odds(
                            sport=sport_key,
                            date=date,
                            event_ids=event_id
                        )

                        # Find matching odds for this event
                        matching_odds = next(
                            (odds for odds in odds_data.get('data', []) 
                             if odds['id'] == event_id), 
                            None
                        )

                        if matching_odds:
                            # Only insert if event doesn't exist
                            if not self.events_collection.find_one({'event_id': event_id}):
                                # Create event document with all available data
                                event_document = {
                                    'event_id': event_id,
                                    'sport_key': matching_odds.get('sport_key'),
                                    'sport_title': matching_odds.get('sport_title'),
                                    'commence_time': matching_odds.get('commence_time'),
                                    'home_team': matching_odds.get('home_team'),
                                    'away_team': matching_odds.get('away_team'),
                                    'historical_odds': [{
                                        'timestamp': events_data.get('timestamp'),
                                        'bookmakers': matching_odds.get('bookmakers', [])
                                    }]
                                }
                                self.events_collection.insert_one(event_document)

                # Move to next timestamp based on last event's commence_time
                if 'data' in events_data and events_data['data']:
                    # Sort events by commence_time and get the latest one
                    latest_event = max(events_data['data'], key=lambda x: x['commence_time'])
                    latest_event_time = datetime.strptime(latest_event['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                    
                    # If current_date is within a day of latest_event_time, add a day
                    # If current_date is greater than latest_event_time, add a week
                    if current_date > latest_event_time:
                        current_date = current_date + timedelta(weeks=1)
                    elif (latest_event_time - current_date).days < 1:
                        current_date = current_date + timedelta(days=1)
                    else:
                        current_date = latest_event_time

                # Respect API rate limits
                time.sleep(1)

            except Exception as e:
                print(f"Error processing date {current_date}: {str(e)}")
                time.sleep(5)  # Wait longer if there's an error
                continue

    def get_stored_odds(self, event_id):
        """Retrieve all historical odds for a specific event."""
        event = self.events_collection.find_one({'event_id': event_id})
        return event.get('historical_odds', []) if event else []
