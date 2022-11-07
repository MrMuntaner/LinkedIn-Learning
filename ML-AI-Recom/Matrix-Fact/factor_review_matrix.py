import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Code below: Load user ratings
raw_dataset_df = pd.read_csv('data/movie_ratings_data_set.csv')

# Code below:
# Convert the running list of user ratings into a matrix
# This will contain a sparse array of the reviews
ratings_df = pd.pivot_table(raw_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)

# Code below:
# Factor the array to find the user attributes matrix and the movie attributes matrix-
# that i can multiply back together to re-create the ratings data

# Apply matrix factorization to find the latent features:
# To do this i'll use the low rank matrix factorization algorithm.
# The matrix_factorization_utilities.py shows the implementation of this.

# First i pass in the rating data, but call pandas as matrix function to make sure i-
# pass them as a numpy matrix data type.
# Next, this method takes in a parameter called num_features, which controls how many-
# latent features to generate for each user and each movie.

# Now the regularization_amount takes in 0.1
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.to_numpy(),
                                                                    num_features=15,
                                                                    regularization_amount=0.1)

# The result of the function is a U matrix and an N matrix that has 15 attributes for each user-
# and each movie respectively. 

# Code below:
# Now i can get the ratings for every movie by multiplying U and N together. But instead-
# of using the regular multiplication operator, i'll use numpys matmul function so that-
# it knows i want to do matrix multiplication.

# Find all predicted ratings by multiplying the U by M and store them in the array called predicted_ratings
predicted_ratings = np.matmul(U, M)

# Code below:
# Save all the ratings to a csv file:
# First, i'll create a new pandas dataframe to hold the data.
# For this dataframe i'll tell pandas to use the same row and column names as i have in the ratings_df dataframe
predicted_ratings_df = pd.DataFrame(index=ratings_df.index,
                                    columns=ratings_df.columns,
                                    data=predicted_ratings)
predicted_ratings_df.to_csv("predicted_ratings.csv")

# When looking at the predicted_ratings.csv in Excel, the data looks just like the original review data,-
# except now every cell is filled in.

# I now have an estimate for how many every single user would rate every single movie.
# For example, i can see with user 3 rating movie 4 that they would give it a rating of about 4 stars. 

# Now that i know all these ratings i can start recommending movies to users in the order of their score.

# Looking at user 1 lets see which movie i'd recommend to them. 
# Of all these movies, if i exclude the ones the user had previously rated, movie number 34 is the one with the highest score.

# Thats the first movie i should recommend to this user. When the user watches this movie, i'll ask them to rate it.
# If their rating disagrees with what i predicted, i'll add that new rating in and recalculate this matrix.
# That will improve the my overall ratings.

# The more ratings i have to work from, the less holes i'll have in my ratings array and the better chance i'll have-
# of coming up with more accurate values for the U and M matrices.