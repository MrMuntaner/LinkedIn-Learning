import pickle
import pandas as pd

# This file contains just the logic to recommend movies using the data files created.

# Load prediction rules from data files
U = pickle.load(open("Real-World-Program\user_features.dat" "rb"))
M = pickle.load(open("Real-World-Program\product_features.dat" "rb"))
predicted_ratings = pickle.load(open("Real-World-Program\predicted_ratings.dat" "rb"))

# Load movie titles beacuse i'll want to have access to the movie titles 
movies_df = pd.read_csv('data/movies.csv', index_col='movie_id')

print("Enter a user_id to get recommendations (Between 1 and 100):")
user_id_to_search = int(input())

print("Movies we will recommend:")

user_ratings = predicted_ratings[user_id_to_search - 1]
movies_df['rating'] = user_ratings
movies_df = movies_df.sort_values(by=['rating'], ascending=False)

print(movies_df[['title', 'genre', 'rating']].head(5))