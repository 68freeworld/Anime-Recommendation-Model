from flask import Flask, send_from_directory, jsonify, request
import pandas as pd
from models import hybrid_npr, HybridContent, load_data
import os
import requests

# Load data once at startup
rating_df_clean, anime_df_clean = load_data()

app = Flask(__name__, static_folder='frontend/build', static_url_path='')

# Simple in-memory cache for poster URLs
_poster_cache = {}

def fetch_poster(anime_id):
    """Retrieve poster image from Jikan API and cache the result."""
    if anime_id in _poster_cache:
        return _poster_cache[anime_id]
    try:
        resp = requests.get(
            f"https://api.jikan.moe/v4/anime/{anime_id}", timeout=5,
            headers={"User-Agent": "anime-recommender"}
        )
        if resp.status_code == 200:
            data = resp.json()
            url = data.get("data", {}).get("images", {}).get("jpg", {}).get("image_url", "")
        else:
            url = ""
    except Exception:
        url = ""
    if not url:
        url = "https://via.placeholder.com/150x210?text=No+Image"
    _poster_cache[anime_id] = url
    return url

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/top-anime', methods=['GET'])
def get_top_anime():
    top_recs = hybrid_npr(rating_df_clean, anime_df_clean)
    records = top_recs.to_dict(orient='records')
    for rec in records:
        rec['image_url'] = fetch_poster(rec['anime_id'])
    return jsonify(records)


@app.route('/api/recommend', methods=['GET'])
def recommend():
    """Return recommendations based on a provided anime title."""
    title = request.args.get('title', '')
    if not title:
        return jsonify([])
    try:
        recs = HybridContent(title, anime_df_clean)
    except Exception:
        return jsonify([])

    # recs may be a Series of anime names
    if hasattr(recs, 'to_frame'):
        recs = recs.to_frame(name='anime_name')
    if 'anime_name' not in recs.columns:
        recs = recs.rename(columns={recs.columns[0]: 'anime_name'})

    result = []
    for name in recs['anime_name']:
        row = anime_df_clean.loc[anime_df_clean['anime_name'] == name]
        if row.empty:
            continue
        aid = int(row.iloc[0]['anime_id'])
        result.append({
            'anime_name': name,
            'anime_id': aid,
            'image_url': fetch_poster(aid)
        })
    return jsonify(result)

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
