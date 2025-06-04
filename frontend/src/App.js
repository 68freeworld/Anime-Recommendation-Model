import React, { useEffect, useState } from 'react';
import './App.css';
import AnimeList from './AnimeList';

function App() {
  const [recommendations, setRecommendations] = useState([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    fetch('/api/top-anime')
      .then(res => res.json())
      .then(data => setRecommendations(data))
      .catch(err => console.error('Failed to fetch recommendations', err));
  }, []);

  const handleSearch = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    fetch(`/api/recommend?title=${encodeURIComponent(query)}`)
      .then(res => res.json())
      .then(data => setResults(data))
      .catch(err => console.error('Failed to fetch search', err));
  };

  return (
    <div className="App">
      <header>
        <h1>Anime Recommendations</h1>
        <form className="search" onSubmit={handleSearch}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search anime title"
          />
          <button type="submit">Search</button>
        </form>
      </header>

      <main>
        {results.length > 0 && (
          <AnimeList title={`Results for "${query}"`} list={results} />
        )}
        {query && results.length === 0 && (
          <p className="no-results">No results found.</p>
        )}

        <AnimeList title="Top Anime" list={recommendations} />
      </main>
    </div>
  );
}

export default App;
