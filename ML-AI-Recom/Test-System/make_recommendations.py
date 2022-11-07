import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Load user ratings
raw_dataset_df = pd.read_csv('data/movie_ratings_data_set.csv')

# Load movie titles
movies_df = pd.read_csv('data/movies.csv', index_col='movie_id')

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(raw_dataset_df, index='user_id',
                            columns='movie_id',
                            aggfunc=np.max)

# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.to_numpy(),
                                                                    num_features=15,
                                                                    regularization_amount=0.1)

# Find all predicted ratings for each user by multiplying U and M matrices
# Now that I have predicted ratings, I can make predictions.
predicted_ratings = np.matmul(U, M)

# Prompt for a user id, that way the user can type in any user ID to see recommendations-
# for different users.
print("Enter a user_id to get recommendations (Between 1 and 100): ")
user_id_to_search = int(input())

# Print out the movies the user has already rated themself.
print("Movies previously reviewed by user_id {}:".format(user_id_to_search))

# I can look them up in the raw data set df dataframe.
# Pandas will filter down the list to the entries where the user ID is the same one as-
# the user just typed in.
reviewed_movies_df = raw_dataset_df[raw_dataset_df['user_id'] == user_id_to_search]

# Next, i'll join this list of reviews with the movie df dataframe so i can disply the title-
# of each movie.
reviewed_movies_df = reviewed_movies_df.join(movies_df, on='movie_id')


print(reviewed_movies_df[['title', 'genre', 'value']])

input("Press enter to continue. ")

print("Movies we will recommend: ")

# First i will pull out the predicted ratings for this specific user from the predicted-
# ratings array.
user_ratings = predicted_ratings[user_id_to_search -1]

# Now i can save the predicted rating for each movie back to the list of movies to make it easy to-
# print out.
movies_df['rating'] = user_ratings

# At this point i have a list of movies with a score for every movie based on how much the user would-
# like it, but i dont want to show the movies that the user has already rated. 

# So i need to exclude those movies from the list.
# I can get the list of the movies that the user has already reviewed and save those to a variable-
# called already_revied 
already_reviewed = reviewed_movies_df['movie_id']

# looking at the movies that are not in that list
# This line is a little complex.
# First i am using the isin function to find the movies that are in the list-
# then i'm comparing that list with false to invert it and find the movies that are not in that list
recommended_df = movies_df[movies_df.index.isin(already_reviewed) == False]

# Then i can use sort values to sort the list, so that the highest rated movie is first.
recommended_df = recommended_df.sort_values(by=['rating'], ascending=False)


# Finally, I can use pandas head function to print out the first 5 movies in the list.
print(recommended_df[['title', 'genre', 'rating']].head(5))

# Run program:
# When asked for the user id, i put 2