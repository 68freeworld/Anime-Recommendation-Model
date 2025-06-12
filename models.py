import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data into TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity  # For computing cosine similarity between vectors
from scipy.spatial.distance import pdist, squareform  # For pairwise distance computations and converting to a square matrix
import pickle
from sklearn.preprocessing import MinMaxScaler
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ---------- data loader ----------
def load_data(rating_path="rating_df_final.pk", anime_path="anime_df_final.pk"):
    """Load and return the rating and anime data frames."""
    with open(rating_path, "rb") as f:
        rating_df = pickle.load(f)
    with open(anime_path, "rb") as f:
        anime_df = pickle.load(f)
    return rating_df, anime_df

from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity
# from tqdm.notebook import tqdm


# placeholders for precomputed data
user2movie = None
movie2user = None
user_movie2rating = None

uid2idx = None
iid2idx = None
idx2iid = None

R = None
user_mean = None
item_mean = None
global_mean = None

user_sim = None
item_sim = None



def precomps(rating_df, K=50):
    global user2movie, movie2user, user_movie2rating
    global uid2idx, iid2idx, idx2iid
    global R, user_mean, item_mean, global_mean
    global user_sim, item_sim

    # 1. helper dicts
    user2movie        = rating_df.groupby("user_id")["anime_id"].apply(list).to_dict()
    movie2user        = rating_df.groupby("anime_id")["user_id"].apply(list).to_dict()
    user_movie2rating = {
        (r.user_id, r.anime_id): r.rating
        for r in rating_df.itertuples()
    }

    # 2. build sparse USER×ITEM matrix
    uid2idx = {u:i for i,u in enumerate(rating_df["user_id"].unique())}
    iid2idx = {i:j for j,i in enumerate(rating_df["anime_id"].unique())}
    idx2iid = {j:i for i,j in iid2idx.items()}

    rows = rating_df["user_id"].map(uid2idx)
    cols = rating_df["anime_id"].map(iid2idx)
    data = rating_df["rating"].astype(float)
    R = csr_matrix((data, (rows, cols)),
                   shape=(len(uid2idx), len(iid2idx)))

    user_mean   = (R.sum(1).A1) / ((R != 0).sum(1).A1)
    item_mean   = (R.sum(0).A1) / ((R != 0).sum(0).A1)
    global_mean = data.mean()

    # 3. precompute top-K sparse similarities
    user_sim = topk_cosine(R,   K=K)
    item_sim = topk_cosine(R.T, K=K)




# ---------------- 3. cosine similarity (top-K pruning) -------

def topk_cosine(mat, K=50):
    # compute full (sparse) cosine‐similarity matrix
    sim = cosine_similarity(mat, dense_output=False)   # CSR × CSR
    trimmed = []
    # loop over every row i
    for i in range(sim.shape[0]):
        row = sim.getrow(i).tocoo()
        # pick the top‐K most similar entries in that row
        idx = row.data.argsort()[::-1][:K]
        trimmed.append(
            csr_matrix(
                (row.data[idx], (np.zeros_like(idx), row.col[idx])),
                shape=(1, sim.shape[1])
            )
        )
    # stack all trimmed rows back into a sparse matrix
    return vstack(trimmed).tocsr()


# ---------------- 4. rating predictors -----------------------
def predict_user_cf(u, i, sim_cut=0.05):
    if (u not in uid2idx) or (i not in iid2idx): return global_mean
    ui, ii = uid2idx[u], iid2idx[i]
    sims   = user_sim.getrow(ui).tocoo()
    num=den=0.0
    for s,v in zip(sims.data, sims.col):
        if s < sim_cut: break
        r_vi = R[v, ii]
        if r_vi==0: continue
        num += s * (r_vi - user_mean[v])
        den += abs(s)
    return user_mean[ui] + num/den if den else user_mean[ui]

def predict_item_cf(u, i, sim_cut=0.05):

    if (u not in uid2idx) or (i not in iid2idx): return global_mean
    ui, ii = uid2idx[u], iid2idx[i]
    sims   = item_sim.getrow(ii).tocoo()
    num=den=0.0
    for s,j in zip(sims.data, sims.col):
        if s < sim_cut: break
        r_uj = R[ui, j]
        if r_uj==0: continue
        num += s * (r_uj - item_mean[j])
        den += abs(s)
    return item_mean[ii] + num/den if den else item_mean[ii]

# ---------------- 5. recommendation helpers -----------------
def uCF_list(user, top_n=500):
    seen = set(user2movie.get(user, []))
    cand = [m for m in movie2user if m not in seen]
    scores = {m: predict_user_cf(user, m) for m in cand}
    return sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_n]

def itemCF_list(user, top_n=500):
    seen = set(user2movie.get(user, []))
    cand = [m for m in movie2user if m not in seen]
    scores = {m: predict_item_cf(user, m) for m in cand}
    return sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_n]

# ---------------- 6. HYBRID recommender ----------------------
def HybridCF(user, rating_df, top_n=10, alpha=0.4):
    precomps(rating_df)
    u_preds = dict(uCF_list(user, top_n=1000))
    i_preds = dict(itemCF_list(user, top_n=1000))
    movies = set(u_preds) | set(i_preds)
    hybrid = {}

    # Combine predictions into a hybrid score
    for m in movies:
        if m in u_preds and m in i_preds:
            hybrid[m] = alpha * u_preds[m] + (1 - alpha) * i_preds[m]
        elif m in u_preds:
            hybrid[m] = u_preds[m]
        else:
            hybrid[m] = i_preds[m]

    # Create a DataFrame from the hybrid scores
    recs = pd.DataFrame(sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n],
                        columns=['anime_id', 'Hybrid_Score'])

    # Merge with anime names
    id_name = rating_df[['anime_id', 'anime_name']].drop_duplicates()
    recs = recs.merge(id_name, on='anime_id', how='left')

    # Select and reorder columns, reset the index without adding an index column
    recs = recs[['anime_name', 'anime_id']]
    recs = recs.reset_index(drop=True)
    
    return recs


# =============================================================
# 1. HELPER FUNCTIONS
# =============================================================
from sklearn.preprocessing import MinMaxScaler

def hybrid_npr(rating_df_clean=None, anime_df_clean=None, top_n=10, alpha=0.5):
    if rating_df_clean is None or anime_df_clean is None:
        rating_df_clean, anime_df_clean = load_data()
    # Aggregate and merge data
    anime_stats = (
        rating_df_clean
        .groupby("anime_id")
        .agg(Num_Ratings=("rating", "size"),
             Avg_Rating=("rating", "mean"))
        .reset_index()
        .merge(anime_df_clean[["anime_id", "anime_name"]], on="anime_id")
    )

    # Compute weighted and Bayesian scores
    anime_stats = add_weighted_score(anime_stats, w1=0.5, w2=0.5)
    anime_stats = add_bayesian_score(anime_stats, rating_df_clean)

    # Normalize scores
    scaler = MinMaxScaler()
    anime_stats[["Weighted_norm", "Bayesian_norm"]] = scaler.fit_transform(
        anime_stats[["Weighted_Score", "Bayesian_Rating"]]
    )

    # Calculate hybrid score
    anime_stats["Hybrid_Score"] = (
        alpha * anime_stats["Bayesian_norm"] +
        (1 - alpha) * anime_stats["Weighted_norm"]
    )

    # Select top N recommendations
    top_recs = (
        anime_stats
        .sort_values("Hybrid_Score", ascending=False)
        .loc[:, ["anime_name", "anime_id"]]
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_recs

def add_weighted_score(df, w1=0.5, w2=0.5):
    max_n = df["Num_Ratings"].max()
    max_r = df["Avg_Rating"].max()
    df["Norm_Num_Ratings"] = df["Num_Ratings"] / max_n
    df["Norm_Avg_Rating"] = df["Avg_Rating"] / max_r
    df["Weighted_Score"] = w1 * df["Norm_Num_Ratings"] + w2 * df["Norm_Avg_Rating"]
    return df

def add_bayesian_score(df, ratings_df):
    C = df["Num_Ratings"].mean()
    m = ratings_df["rating"].mean()
    df["Bayesian_Rating"] = (C * m + df["Num_Ratings"] * df["Avg_Rating"]) / (C + df["Num_Ratings"])
    return df
    # If you literally need the list object:
    # return top_names   # inside a function




def HybridContent(title: str,
                  anime_df: pd.DataFrame = None,
                  top_n: int = 10,
                  alpha: float = 0.7) -> pd.DataFrame:
    """
    Hybrid content-based recommender that blends TF-IDF synopsis similarity
    with Jaccard genre similarity.

    Parameters
    ----------
    title      : str
        Exact anime title (anime_name) to base recommendations on.
    anime_df   : pd.DataFrame
        Must contain columns ['anime_name', 'genre', 'synopsis'].
        - 'genre' should be a string like "Action, Fantasy, Shounen".
    top_n      : int, default 10
        Number of recommendations to return.
    alpha      : float, default 0.7
        Weight for TF-IDF similarity.  (1-alpha) is weight for Jaccard.

    Returns
    -------
    pd.DataFrame  (top_n rows)
        Columns: ['anime_name', 'TFIDF', 'Jaccard', 'Hybrid']
    """
    if anime_df is None:
        _, anime_df = load_data()
    # ------------------------------------------------------------
    # 0. basic checks
    # ------------------------------------------------------------
    if title not in anime_df['anime_name'].values:
        raise ValueError(f"'{title}' not found in anime_df['anime_name'].")

    df = anime_df[['anime_name', 'Genres', 'Synopsis']].dropna(subset=['Synopsis'])

    # ------------------------------------------------------------
    # 1. TF-IDF similarity on synopsis
    # ------------------------------------------------------------
    tfidf = TfidfVectorizer(stop_words='english')
    tf_mat = tfidf.fit_transform(df['Synopsis'])
    idx_map = {name: i for i, name in enumerate(df['anime_name'])}
    base_vec = tf_mat[idx_map[title]]
    tf_scores = cosine_similarity(base_vec, tf_mat).flatten()
    tf_ser = pd.Series(tf_scores, index=df['anime_name'], name='TFIDF')

    # ------------------------------------------------------------
    # 2. Jaccard similarity on genre tokens
    # ------------------------------------------------------------
    def to_set(genres: str) -> set:
        return {g.strip().lower() for g in genres.split(',') if g.strip()}

    base_genres = to_set(df.loc[idx_map[title], 'Genres'])
    jac_scores = []
    for g in df['Genres']:
        other = to_set(g)
        inter = len(base_genres & other)
        union = len(base_genres | other)
        jac_scores.append(inter / union if union else 0.0)
    jac_ser = pd.Series(jac_scores, index=df['anime_name'], name='Jaccard')

    # ------------------------------------------------------------
    # 3. scale (0-1) and blend
    # ------------------------------------------------------------
    scaler = MinMaxScaler()
    tf_scaled  = pd.Series(scaler.fit_transform(tf_ser.values.reshape(-1, 1)).flatten(),
                           index=tf_ser.index)
    jac_scaled = pd.Series(scaler.fit_transform(jac_ser.values.reshape(-1, 1)).flatten(),
                           index=jac_ser.index)

    hybrid = alpha * tf_scaled + (1 - alpha) * jac_scaled
    hybrid.name = 'Hybrid'

    # ------------------------------------------------------------
    # 4. build result table
    # ------------------------------------------------------------
    recs = (pd.concat([tf_ser, jac_ser, hybrid], axis=1)
              .drop(index=title)
              .sort_values('Hybrid', ascending=False)
              .head(top_n)
              .reset_index())
    recs.rename(columns={'index': 'anime_name'})
    recs=recs["anime_name"]
    return recs



# 1. Define the function (with return statement)
def addrows(user_id, watched, rating_df_clean):
    new_rows = []
    for name in watched:
        # find the anime_id
        match = rating_df_clean.loc[
            rating_df_clean['anime_name'].str.casefold() == name.casefold()
        ]
        if match.empty:
            print(f" '{name}' not found — skipping")
            continue

        aid = int(match.iloc[0]['anime_id'])

        # check for an existing rating by this user
        already = rating_df_clean[
            (rating_df_clean['user_id'] == user_id) &
            (rating_df_clean['anime_id'] == aid)
        ]
        if not already.empty:
            print(f" You have already rated '{name}' — skipping")
            continue

        # ask for a new rating
        while True:
            try:
                rating = float(input(f"Your rating for '{name}' (1–10): "))
                if 1 <= rating <= 10:
                    break
                print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Not a number — try again.")

        new_rows.append({
            "user_id":   user_id,
            "anime_id":  aid,
            "rating":    rating,
            "anime_name": name
        })

    # append all at once and return
    if new_rows:
        rating_df_clean = pd.concat(
            [rating_df_clean, pd.DataFrame(new_rows)],
            ignore_index=True
        )
        print(f"✅ Added {len(new_rows)} new ratings for user {user_id}")
    else:
        print("No new ratings added.")

    return rating_df_clean



        