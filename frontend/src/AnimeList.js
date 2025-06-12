import React from 'react';

const PLACEHOLDER = 'https://media.giphy.com/media/14uQ3cOFteDaU/giphy.gif';

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
              src={anime.image_url || PLACEHOLDER}
              alt={anime.anime_name}
              loading="lazy"
              onError={(e) => {
                e.currentTarget.src = PLACEHOLDER;
              }}
            />
            {anime.anime_name}
          </li>
        ))}
      </ul>
    </section>
  );
}

export default AnimeList;
