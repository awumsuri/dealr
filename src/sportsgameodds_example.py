from api.sport_game_odds_api import SportsGameOddsAPI

next_cursor = None
event_data = []
api = SportsGameOddsAPI()

while True:
    params = {
        'leagueID': 'NBA',
        'marketOddsAvailable': "true",
        'limit': 50,
        'cursor': next_cursor
    }
    
    data = api.get(params, 'events')
    if not data:
        break
        
    event_data.extend(data['data'])        
    next_cursor = data.get('nextCursor')
    
    if not next_cursor:
        events = []
        for event in event_data:
            event_id = event['eventID']
            params = {
                'eventID': event_id,
                'startsBefore': '2025-01-01',
                'marketOddsAvailable': "true"
            }
            data = api.get(params, 'events')
            print(data)
            break
        break

for event in event_data:
    odds = event['odds']
    for odd_id, odd_object in odds.items():
        print(f'Odd ID: {odd_id}')
        print(f'Odd Value: {odd_object["closeOdds"]}')

      
