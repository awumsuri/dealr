from odds_api import OddsAPI
from datetime import datetime, timedelta
from typing import Dict, List, Any
from constants.sports_types import SoccerUefaEuropeanChampionship

class DataGatherer:
    def __init__(self):
        self.api = OddsAPI()
    
    def get_historical_match_data(self, team1: str, team2: str, sport: str = SoccerUefaEuropeanChampionship) -> Dict[str, Any]:
        """
        Gather historical match data and odds for matches involving either team.
        
        Args:
            team1: First team name
            team2: Second team name
            sport: Sport key (default: "soccer")
            
        Returns:
            Dictionary containing historical match data and odds
        """
        # Get historical odds for both teams
        # Using last 30 days as example timeframe
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        team1_history = self.api.get_historical_odds(
            sport=sport,
            team=team1,
            from_date=start_date,
            to_date=end_date
        )
        
        team2_history = self.api.get_historical_odds(
            sport=sport,
            team=team2,
            from_date=start_date,
            to_date=end_date
        )
        
        # Get scores for the matches and filter for relevant teams
        all_scores = self.api.get_scores(sport_key=sport, daysFrom=30)
        relevant_scores = [
            score for score in all_scores
            if team1 in (score.get('home_team'), score.get('away_team')) or
               team2 in (score.get('home_team'), score.get('away_team'))
        ]
        
        return {
            "team1_history": team1_history,
            "team2_history": team2_history,
            "recent_scores": relevant_scores
        }
    
    def process_match_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the raw match data into a structured format.
        
        Args:
            data: Raw match data from get_historical_match_data
            
        Returns:
            List of processed match data with odds and results
        """
        processed_matches = []
        
        # Process team histories and match them with scores
        # This is a placeholder - implement the actual processing logic
        # based on the API response structure
        
        return processed_matches

def main():
    gatherer = DataGatherer()
    # Example usage
    match_data = gatherer.get_historical_match_data(
        team1="Manchester United",
        team2="Manchester City"
    )
    processed_data = gatherer.process_match_data(match_data)
    print(processed_data)

if __name__ == "__main__":
    main()
