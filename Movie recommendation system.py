import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load movie data
data = pd.read_csv("movies.csv")

# Define features
features = ["title", "overview", "genres"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform features into TF-IDF vectors
vectors = vectorizer.fit_transform(data[features].values.flatten())

def cosine_similarity(movie1, movie2):
  """
  Calculates the cosine similarity between two movies.
  """
  return vectors.dot(movie1, movie2) / (np.linalg.norm(movie1) * np.linalg.norm(movie2))

# Function to get recommendations for a given movie
def get_recommendations(movie_id, k=5):
  """
  Recommends k movies similar to the given movie based on cosine similarity.
  """
  movie_vector = vectors[movie_id]
  similarities = [cosine_similarity(movie_vector, v) for v in vectors]
  top_k_indices = np.argsort(similarities)[-k:]
  recommended_movies = data.iloc[top_k_indices]
  return recommended_movies

# Get movie ID from user input or another source
movie_id = 123

# Get recommendations
recommended_movies = get_recommendations(movie_id)

# Print recommendations
print("Recommended movies for", data.loc[movie_id, "title"] + ":")
for i, movie in recommended_movies.iterrows():
  print(f"\t- {movie['title']}")

