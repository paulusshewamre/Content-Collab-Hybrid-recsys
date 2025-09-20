import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ["Inception", "Interstellar", "The Dark Knight", "Avengers", "Guardians of the Galaxy"],
    'genres': ["sci-fi action", "sci-fi drama", "action crime", "action fantasy", "sci-fi fantasy"]
}
df = pd.DataFrame(data)

# Vectorize genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_content(movie_title):
    idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended = [df['title'][i[0]] for i in scores[1:4]]
    return recommended

print("Content-based:", recommend_content("Inception"))
