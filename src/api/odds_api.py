import requests
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import os
from datetime import datetime

class OddsAPI:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through ODDS_API_KEY environment variable")
        self.base_url = "https://api.the-odds-api.com/v4"
    
    def get_sports(self) -> List[Dict[str, Any]]:
        """Get a list of all available sports."""
        endpoint = f"{self.base_url}/sports"
        params = {"apiKey": self.api_key}
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_odds(
        self,
        sport_key: str,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "decimal",
        date_format: str = "iso"
    ) -> List[Dict[str, Any]]:
        """
        Get odds for a specific sport.
        
        Args:
            sport_key: The sport key (e.g., 'americanfootball_nfl')
            regions: Comma-separated list of regions (e.g., 'us,uk,eu')
            markets: Comma-separated list of markets (e.g., 'h2h,spreads')
            odds_format: Format of odds ('decimal' or 'american')
            date_format: Format of dates ('iso' or 'unix')
        """
        endpoint = f"{self.base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_scores(
        self,
        sport_key: str,
        daysFrom: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get scores for a specific sport.
        
        Args:
            sport_key: The sport key (e.g., 'americanfootball_nfl')
            daysFrom: Number of days from which to get historical scores
        """
        endpoint = f"{self.base_url}/sports/{sport_key}/scores"
        params = {"apiKey": self.api_key}
        if daysFrom is not None:
            params["daysFrom"] = daysFrom
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_event_odds(
        self,
        sport_key: str,
        event_id: str,
        markets: str = "h2h",
        regions: str = "us"
    ) -> Dict[str, Any]:
        """
        Get odds for a specific event.
        
        Args:
            sport_key: The sport key (e.g., 'americanfootball_nfl')
            event_id: The specific event ID
            markets: Comma-separated list of markets (e.g., 'h2h,spreads')
            regions: Comma-separated list of regions (e.g., 'us,uk,eu')
        """
        endpoint = f"{self.base_url}/sports/{sport_key}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_historical_odds(
        self,
        sport: str,      
        date: Optional[str] = None,
        event_ids: Optional[str] = None,
        regions: str = "eu,us"       
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/historical/sports/{sport}/odds"
        
        params = {
            'apiKey': self.api_key,           
            'date': date,
            'regions': regions,
            'eventIds': event_ids
        }       
       

        response = requests.get(endpoint, params=params)
        return response.json()
    
    def get_events(
        self,
        sport_key: str,
        date_format: str = "iso",
        event_ids: Optional[str] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a list of in-play and pre-match events for a specified sport.
        This endpoint does not count against the usage quota.
        
        Args:
            sport_key: The sport key (e.g., 'americanfootball_nfl')
            date_format: Format of dates ('iso' or 'unix')
            event_ids: Optional comma-separated game ids to filter specific events
            commence_time_from: Optional
            ISO 8601 datetime to filter events starting from
                              (e.g., '2023-09-09T00:00:00Z')
            commence_time_to: Optional ISO 8601 datetime to filter events until
                            (e.g., '2023-09-10T23:59:59Z')
        """
        endpoint = f"{self.base_url}/sports/{sport_key}/events"
        params = {
            "apiKey": self.api_key,
            "dateFormat": date_format
        }
        
        if event_ids:
            params["eventIds"] = event_ids
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to

        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def find_event_ids(
        self,
        sport_key: str,
        team1: str,
        team2: str,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None
    ) -> List[str]:
        """
        Find all event IDs for matches between two specific teams.
        
        Args:
            sport_key: The sport key (e.g., 'americanfootball_nfl')
            team1: Name of the first team
            team2: Name of the second team
            commence_time_from: Optional ISO 8601 datetime to filter events starting from
            commence_time_to: Optional ISO 8601 datetime to filter events until
        
        Returns:
            List[str]: List of event IDs found, empty list if no matching events are found
        """
        try:
            events = self.get_historical_events(
                sport=sport_key,
                commence_time_from=commence_time_from,
                commence_time_to=commence_time_to
            )
            
            # Normalize team names for comparison
            team1_lower = team1.lower().strip()
            team2_lower = team2.lower().strip()
            
            matching_events = []
            data = events.get('data', [])
            
            for event in data:
                # Skip if event doesn't have required fields
                if not all(key in event for key in ['id', 'home_team', 'away_team']):
                    continue
                    
                home_team = event['home_team'].lower().strip()
                away_team = event['away_team'].lower().strip()
                
                # Check if both teams are in the event (in either order)
                if ((team1_lower in home_team or team1_lower in away_team) and 
                    (team2_lower in home_team or team2_lower in away_team)):
                    matching_events.append(event['id'])
            
            return matching_events
            
        except Exception as e:
            print(f"Error finding event IDs: {str(e)}")
            return []
    
    def get_historical_events(
        self,
        sport: str,
        regions: str = "eu,us",
        date: Optional[str] = None,
       
    ) -> List[Dict[str, Any]]:
        """
        Get historical events data for a specific sport and date.
        
        Args:
            sport: The sport key (e.g., 'americanfootball_nfl')
            date: Timestamp of the data snapshot in ISO8601 format (e.g., '2023-10-10T12:15:00Z').
                  Defaults to current time if not provided.
            date_format: Format of dates ('iso' or 'unix')
            event_ids: Optional comma-separated game ids to filter specific events
            commence_time_from: Optional ISO 8601 datetime to filter events starting from
            commence_time_to: Optional ISO 8601 datetime to filter events until
        
        Returns:
            List of historical events matching the specified criteria
        """
        endpoint = f"{self.base_url}/historical/sports/{sport}/events"
        params = {
            "apiKey": self.api_key,
            "date": date,
            "regions": regions,
        }
      

        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()