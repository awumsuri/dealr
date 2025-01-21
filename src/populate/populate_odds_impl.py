from constants.sports_types import SoccerUefaEuropeanChampionship
from populate_odds import OddsPopulator


def main():
    populate_odds = OddsPopulator()
    populate_odds.populate_historical_data()
    # Example usage
   

if __name__ == "__main__":
    main()
