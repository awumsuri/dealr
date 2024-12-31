from odds_api import OddsAPI
from datetime import datetime, timedelta
from constants.sports_types import SoccerEnglandLeague1

odds_api = OddsAPI()

# Calculate dates for one year range
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

event_id = odds_api.find_event_id(
    sport_key='americanfootball_nfl',
    team1='Kansas City Chiefs',
    team2='San Francisco 49ers',
    commence_time_from=start_date.strftime('%Y-%m-%d'),
    commence_time_to=end_date.strftime('%Y-%m-%d')
)
if event_id:
    print(f"Found event ID: {event_id}")
else:
    print("No matching event found")
