from flask import Flask, send_from_directory, jsonify
import pandas as pd
from models import hybrid_npr  # Make sure this is in models.py
import os

# Load data
anime_df_clean = pd.read_pickle("anime_df_clean.pkl")
rating_df_clean = pd.read_pickle("rating_df_clean.pkl")

app = Flask(__name__, static_folder='frontend/build', static_url_path='')

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/top-anime', methods=['GET'])
def get_top_anime():
    top_recs = hybrid_npr(rating_df_clean, anime_df_clean)
    return jsonify(top_recs.to_dict(orient='records'))

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
