const axios = require('axios');

async function fetchNBAEvents() {
  let nextCursor = null;
  let eventData = [];
  do {
    try {
      const response = await axios.get('https://api.sportsgameodds.com/v1/events', {
        params: {
          leagueID: 'NBA',         
          limit: 50,
          marketOddsAvailable: true,
          cursor: nextCursor
        },
        headers: {
          'X-Api-Key': process.env.SPORTS_GAME_ODDS_API_KEY,
          'Accept': 'application/json'
        }
      });

      const data = response.data;
      eventData = eventData.concat(data.events);
      nextCursor = data?.nextCursor;

    } catch (error) {
      console.error('Error fetching events:', error);
      break;
    }
  } while (nextCursor);

  // Process the events
  eventData.forEach((event) => {
    const odds = event.odds;
    Object.values(odds).forEach((oddObject) => {
      console.log(`Odd ID: ${oddObject.oddID}`);
      console.log(`Odd Value: ${oddObject.closeOdds}`);
    });
  });

  return eventData;
}

fetchNBAEvents();