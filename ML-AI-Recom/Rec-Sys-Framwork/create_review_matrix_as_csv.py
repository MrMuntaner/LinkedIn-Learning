import pandas as pd
import numpy as np

# This code will generate a csv representation of the review matrix that i can open in a spread-
# sheet. 

# Read the dataset into a data table using Pandas
df = pd.read_csv("data/movie_ratings_data_set.csv")

# Convert the running list of user ratings into a matrix using the 'pivot table' function
ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', aggfunc=np.max)

# Create a csv file of the data for easy viewing
ratings_df.to_csv("review_matrix.csv", na_rep="")

# Imagine if i could figure out a way to fill in all the blank spaces based on the numbers i know.

# For example, lets look at user number three. I can see that user number three gave 4 stars to-
# movie 1 and 2, and 5 stars to movie 3.

# What if i could use the ratings i know and the ratings from other users to fill in what what this-
# user would most likely rate movie number 4?

# Once i know the rating that a user would give a movie, i know whether or not i should recommend that movie.
# If i think this user would give movie number 4 a 5 star rating, this is a movie i definitely want to-
# recommend to that user. 

# In order to build a recommendation system, what i really need is an algorithm that helps me complete all the missing-
# blanks in the matrix based on the numbers i already know.

# If i can fill in every blank in the matrix with the rating the user would have given that movie, then i'll know-
# everything i need to know to make a recommendation to every user. 