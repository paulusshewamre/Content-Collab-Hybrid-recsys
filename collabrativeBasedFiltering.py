from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Sample rating dataset
ratings_dict = {
    "itemID": [1, 1, 1, 2, 2, 3, 4, 5],
    "userID": ["A", "B", "C", "A", "C", "B", "A", "C"],
    "rating": [5, 4, 4, 5, 3, 4, 2, 5],
}
df = pd.DataFrame(ratings_dict)

# Load into Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.25)

# Use SVD (Matrix Factorization)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Predict rating
print("Collaborative Filtering:", model.predict("A", 3))
