import React, { useEffect, useState } from 'react';
import './App.css';
import AnimeList from './AnimeList';
import AnimeDetail from './AnimeDetail';
import { Routes, Route, useNavigate } from 'react-router-dom';

function HomePage() {
  const [recommendations, setRecommendations] = useState([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchProgress, setSearchProgress] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    fetch('/api/top-anime')
      .then(res => res.json())
      .then(data => setRecommendations(data))
      .catch(err => console.error('Failed to fetch recommendations', err));
  }, []);

  const handleSearch = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setResults([]);
    setSearchLoading(true);
    setSearchProgress(0);

    let pct = 0;
    const timer = setInterval(() => {
      pct += 10;
      if (pct < 90) setSearchProgress(pct);
    }, 200);

    fetch(`/api/recommend?title=${encodeURIComponent(query)}`)
      .then(res => res.json())
      .then(data => {
        setResults(data);
        setSearchProgress(100);
      })
      .catch(err => console.error('Failed to fetch search', err))
      .finally(() => {
        clearInterval(timer);
        setSearchLoading(false);
      });
  };

  const goToDetail = (anime) => {
    navigate(`/anime/${encodeURIComponent(anime.anime_name)}`, { state: anime });
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
        {searchLoading && (
          <div className="progress"><div style={{width: `${searchProgress}%`}} /></div>
        )}
        {results.length > 0 && (
          <AnimeList title={`Results for \"${query}\"`} list={results} onSelect={goToDetail} />
        )}
        {query && results.length === 0 && (
          <p className="no-results">No results found.</p>
        )}

        <AnimeList title="Top Anime" list={recommendations} onSelect={goToDetail} />
      </main>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/anime/:title" element={<AnimeDetail />} />
    </Routes>
  );
}

export default App;
