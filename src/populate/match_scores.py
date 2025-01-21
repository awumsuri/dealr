import requests
import json

fallback_team_map = {
    "Wolverhampton Wanderers": "Wolves",    
    "Leicester City": "Leicester",
    "Aston Villa": "Aston Villa",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Crystal Palace": "Crystal Palace",
    "Wigan Athletic": "Wigan",
    "West Bromwich Albion": "West Brom",
    "West Bromwich Albion": "West Brom"
}

class MatchScores:
    def __init__(self):
        self.api_key = "dfad0f1bb3c1e3e45297b2cf8aaf1c31"
        self.url = "https://v3.football.api-sports.io/fixtures/headtohead"
        # Load teams data once during initialization
        with open('src/constants/elp_teams.json') as f:
            self.teams_data = json.load(f)
        
    def get_team_id(self, team_name):
        # First try with the full team name
        for year in self.teams_data:
            for team_obj in self.teams_data[year]['response']:
                if team_obj['team']['name'] == team_name:
                    return team_obj['team']['id']
        
        # If not found and team_name has multiple words, try with just the first word
        if ' ' in team_name:
            first_word = team_name.split(' ')[0]
            for year in self.teams_data:
                for team_obj in self.teams_data[year]['response']:
                    if team_obj['team']['name'].startswith(first_word):
                        return team_obj['team']['id']
            
            # If still not found and has 3 or more terms, try with first two terms
            terms = team_name.split()
            if len(terms) >= 3:
                first_two_terms = ' '.join(terms[:2])
                for year in self.teams_data:
                    for team_obj in self.teams_data[year]['response']:
                        if team_obj['team']['name'].startswith(first_two_terms):
                            return team_obj['team']['id']
                        
        # If still not found, try with fallback_team_map
        if team_name in fallback_team_map:
            return self.get_team_id(fallback_team_map[team_name])
        
        return None

    def get_match_result(self, team, opponent, date):
        # Handle team names with multiple words
       
        team_id = self.get_team_id(team)
        opponent_id = self.get_team_id(opponent)

        params = {
            "h2h": f"{team_id}-{opponent_id}",
            "last": "20"
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
