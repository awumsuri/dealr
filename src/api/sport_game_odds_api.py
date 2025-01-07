import requests
import os

class SportsGameOddsAPI:
    def __init__(self):
        self.base_url = 'https://api.sportsgameodds.com/v1'
        self.api_key = os.getenv('SPORTS_GAME_ODDS_API_KEY')

    def get(self, params, endpoint):
        try:
            response = requests.get(
                f'{self.base_url}/{endpoint}',
                headers={'X-Api-Key': self.api_key},
                params=params
            )
            return response.json()
        except Exception as error:
            print(f'Error fetching events: {error}')
            return None