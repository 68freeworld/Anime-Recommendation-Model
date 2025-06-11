import React, { useEffect, useState } from 'react';
import AnimeList from './AnimeList';
import { useParams, useLocation, useNavigate } from 'react-router-dom';

function AnimeDetail() {
  const { title } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [recs, setRecs] = useState([]);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setProgress(0);
    setLoading(true);
    let pct = 0;
    const timer = setInterval(() => {
      pct += 10;
      if (pct < 90) setProgress(pct);
    }, 200);

    fetch(`/api/recommend?title=${encodeURIComponent(title)}`)
      .then(res => res.json())
      .then(data => {
        setRecs(data);
        setProgress(100);
      })
      .catch(err => console.error('Failed to load recommendations', err))
      .finally(() => {
        clearInterval(timer);
        setLoading(false);
      });

    return () => clearInterval(timer);
  }, [title]);

  const animeName = location.state?.anime_name || title;

  return (
    <div className="App">
      <header>
        <h1>{animeName}</h1>
        <button className="back-button" onClick={() => navigate(-1)}>Back</button>
      </header>
      <main>
        {loading && (
          <div className="progress"><div style={{width: `${progress}%`}} /></div>
        )}
        {!loading && recs.length === 0 && (
          <p className="no-results">No recommendations found.</p>
        )}
        <AnimeList title="Recommended" list={recs} onSelect={(a) => navigate(`/anime/${encodeURIComponent(a.anime_name)}`, { state: a })} />
      </main>
    </div>
  );
}

export default AnimeDetail;
