import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Code below:
# Load user ratings
df = pd.read_csv('data/movie_ratings_data_set.csv')

# Code below:
# Load movie titles
movies_df = pd.read_csv('data/movies.csv', index_col='movie_id')

# Code below:
# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', aggfunc=np.max)

# Code below:
# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.to_numpy(),
                                                                    num_features=15,
                                                                    regularization_amount=1.0)

# Code below:
# Right now, each movie is represented by one column in the M matrix
# Swap the rows and columns of product_features just so it's easier to work with.

# First, i'll use numpy's transpose function to flip flop the matrix so each column becomes-
# a row.
M = np.transpose(M)

# Code below:
# The M matrix has 15 unique values for each movie that represent the characteristics-
# of that movie. This means that other movies with nearly the same numbers should-
# be very similar.

# To find other movies similar to this one, I just need to find the other movies-
# whose numbers are closest to this movie's numbers. It's just a subtraction problem

# Choose a movie to find similar movies to. Let's find movies similar to movie #5:
movie_id = 5

# Get movie #1's name and genre
movie_information = movies_df.loc[5]

print("I am finding movies similar to this movie:")
print("Movie title: {}".format(movie_information.title))
print("Genre: {}".format(movie_information.genre))

# Code below:
# I have to subtract 1 here because M is zero indexed, but the movie ID's start at 1.
# Get the features for movie #1 we found via matrix factorization
current_movie_features = M[movie_id - 1]

# Print out the movie attributes so i can see what they look like.
# With these attributes i am ready to find similar movies. 
print("The attributes for this movie are:")
print(current_movie_features)

# Code below:
# The main logic for finding similar movies:

# 1. Subtract the current movie's features from every other movie's features
# This one line of code subtracts current movie features separately from every row-
# of the M matrix.

# This gives me the difference in scores between the current movie and every other movie-
# in the database
difference = M - current_movie_features

# 2. Take the absolute value of that difference (so all numbers are positive)
# This just makes sure that any negative numbers come out as positive.
absolute_difference = np.abs(difference)

# 3. Each movie has 15 features. Sum those 15 features to get a total 'difference score' for each movie
# Use axis=1 to tell numpy to sum up all the numbers in each row and produce a separate sum for each row. 
total_difference = np.sum(absolute_difference, axis=1)

# 4. Create a new column in the movie list with the difference score for each movie
movies_df['difference_score'] = total_difference

# 5. Sort the movie list by difference score, from least different to most different
sorted_movie_list = movies_df.sort_values('difference_score')

# 6. Print the result, showing the 5 most similar movies to movie_id #1
print("The five most similar movies are:")
print(sorted_movie_list[['title', 'difference_score']][0:5])

# The movie the user is looking at is called The Big City Judge 2. 

# Here are the 5 most similar movies that i found:
# 5            The Big City Judge 2          0.000000
# 10        Surrounded by Zombies 1          1.872042
# 9                     Biker Gangs          2.599334
# 3                   The Sheriff 2          2.695185
# 24           The Big City Judge 3          2.787682

# I can ignore the first because its the same movie the user was watching.
# The next 4 movies are the ones that i would show to the user as similar products.

# Based on their titles, these movies seem like they are probably very similar.
# They all seem to be movies about crime and investigation.