from api.odds_api import OddsAPI
from datetime import datetime, timedelta
from constants.sports_types import SoccerEnglandLeague1

odds_api = OddsAPI()

# Calculate dates for one year range
end_date = datetime.now()
start_date = end_date - timedelta(days=365*4)

event_id = odds_api.find_event_ids(
    sport_key="americanfootball_nfl",
    team1='Portsmouth',
    team2='Peterborough United',
    commence_time_from=start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
    commence_time_to=end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
)
if event_id:
    print(f"Found event ID: {event_id}")
else:
    print("No matching event found")
