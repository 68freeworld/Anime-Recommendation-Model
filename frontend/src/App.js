import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    fetch('/api/top-anime')
      .then(res => res.json())
      .then(data => setRecommendations(data))
      .catch(err => console.error("Failed to fetch recommendations", err));
  }, []);

  return (
    <div className="App">
      <header>
        <h1>Top Anime Recommendations</h1>
        <p>These are the most popular and highest rated anime according to our model.</p>
      </header>

      <main>
        <ul className="anime-list">
          {recommendations.map((anime, index) => (
            <li key={anime.anime_id}>
              <span className="rank">#{index + 1}</span> {anime.anime_name}
            </li>
          ))}
        </ul>
      </main>
    </div>
  );
}

export default App;
