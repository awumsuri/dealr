import requests
import json

class MatchScores:
    def __init__(self):
        self.api_key = "dfad0f1bb3c1e3e45297b2cf8aaf1c31"
        self.url = "https://v3.football.api-sports.io/fixtures/headtohead"
        
    def get_team_id(self, team_name):
        with open('src/constants/elp_teams.json') as f:
            teams_data = json.load(f)
            for year in teams_data:
                for team_obj in teams_data[year]['response']:
                    if team_obj['team']['name'] == team_name:
                        return team_obj['team']['id']
        return None

    def get_match_result(self, team, opponent, date):
        team_id = self.get_team_id(team)
        opponent_id = self.get_team_id(opponent)

        params = {
            "h2h": f"{team_id}-{opponent_id}",
            "last": "15"
        }

        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        }

        response = requests.get(self.url, headers=headers, params=params)
        data = response.json()

        for fixture in data["response"]:
            fixture_date = fixture["fixture"]["date"].split("T")[0]
            if fixture_date.startswith(date[:10]):
                match_data = {
                    "home_team": fixture["teams"]["home"]["name"],
                    "away_team": fixture["teams"]["away"]["name"],
                    "home_goals": fixture["goals"]["home"],
                    "away_goals": fixture["goals"]["away"],
                    "result": self._determine_result(
                        fixture["goals"]["home"],
                        fixture["goals"]["away"]
                    )
                }
                return match_data
        return None

    def _determine_result(self, home_goals, away_goals):
        if home_goals > away_goals:
            return "Win"
        elif home_goals < away_goals:
            return "Loss"
        return "Draw"
