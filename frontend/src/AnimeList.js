import React from 'react';

function AnimeList({ title, list, onSelect }) {
  if (!list || list.length === 0) {
    return null;
  }

  return (
    <section>
      {title && <h2>{title}</h2>}
      <ul className="anime-list">
        {list.map((anime, idx) => (
          <li key={anime.anime_id} onClick={() => onSelect && onSelect(anime)} className={onSelect ? 'clickable' : ''}>
            <span className="rank">#{idx + 1}</span>
            <img
              className="poster"
              src={anime.image_url || 'https://via.placeholder.com/150x210?text=No+Image'}
              alt={anime.anime_name}
              loading="lazy"
            />
            {anime.anime_name}
          </li>
        ))}
      </ul>
    </section>
  );
}

export default AnimeList;
