# Hybrid Recommendation System (Content + Collaborative Filtering)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# -------------------------------
# Step 1: Movie and Ratings Dataset
# -------------------------------
movies = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ["Inception", "Interstellar", "The Dark Knight", "Avengers", "Guardians of the Galaxy"],
    'genres': ["sci-fi action", "sci-fi drama", "action crime", "action fantasy", "sci-fi fantasy"]
}
movies_df = pd.DataFrame(movies)

ratings = {
    "userID": ["A", "B", "C", "A", "C", "B", "A", "C"],
    "itemID": [1, 1, 1, 2, 2, 3, 4, 5],
    "rating": [5, 4, 4, 5, 3, 4, 2, 5],
}
ratings_df = pd.DataFrame(ratings)

# -------------------------------
# Step 2: Content-Based Filtering
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_recommend(movie_title, top_n=3):
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended = [movies_df['title'][i[0]] for i in sim_scores[1:top_n+1]]
    return recommended

# -------------------------------
# Step 3: Collaborative Filtering (SVD)
# -------------------------------
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["userID", "itemID", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.25)
svd_model = SVD()
svd_model.fit(trainset)

def collab_predict(user_id, movie_title):
    movie_id = movies_df[movies_df['title'] == movie_title]['movie_id'].iloc[0]
    pred = svd_model.predict(user_id, movie_id)
    return pred.est

# -------------------------------
# Step 4: Hybrid Recommendation
# -------------------------------
def hybrid_recommend(user_id, movie_title, top_n=3, alpha=0.5):
    content_recs = content_recommend(movie_title, top_n=top_n)
    hybrid_scores = {}
    for m in content_recs:
        content_score = 1  # simple equal weight for content-based
        collab_score = collab_predict(user_id, m)
        hybrid_score = alpha * content_score + (1 - alpha) * collab_score
        hybrid_scores[m] = hybrid_score
    recommended = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)
    return recommended

# -------------------------------
# Step 5: Test the Hybrid System
# -------------------------------
print("Hybrid Recommendations for user A based on 'Inception':")
print(hybrid_recommend("A", "Inception"))
