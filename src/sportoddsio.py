import requests

# Set your API key and endpoint
api_key = "5217121f016a446eb7ebbb117653b6d6"
sport = "mlb"
date = "2024-01-01"  # Date of the games
url = f"https://api.sportsdata.io/v3/{sport}/odds/json/GameOddsByDate/{date}"

# Set headers
headers = {
    "Ocp-Apim-Subscription-Key": api_key
}

# Make the request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print("Betting Odds:")
    for game in data:
        print(f"{game['HomeTeam']} vs {game['AwayTeam']}")
        if game['PregameOdds']:
            odds = game['PregameOdds'][0]
            print(f"MoneyLine - Home: {odds['HomeMoneyLine']}, Away: {odds['AwayMoneyLine']}")
            print(f"Point Spread: {odds['PointSpread']}")
        print()
else:
    print(f"Error: {response.status_code}")
    print(response.text)
