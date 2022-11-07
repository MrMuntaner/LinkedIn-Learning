import pandas as pd
import numpy as np
import os
import webbrowser

# In order to build a recommendation system from the data i have, -
# i want to create a matrix or 2D array that shows which movies have been-
# rated by which users.

# the matrix will have one row for each user and one column for each movie

# Read the dataset into a data table using Pandas
df = pd.read_csv("data/movie_ratings_data_set.csv")

# Convert the running list of user ratings into a matrix using the 'pivot table' function
# aggfunc=np.max tells pandas to use Numpy max function to handle duplicates.

# the max function will return the highest number, so if a single user rated the same movie-
# twice, i'll take the higher rating. 
ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', aggfunc=np.max)

# Create a web page view of the data for easy viewing
html = ratings_df.to_html(na_rep="")

# Save the html to a temporary file
with open("review_matrix.html", "w") as f:
    f.write(html)

# Open the web page in our web browser
full_filename = os.path.abspath("review_matrix.html")
webbrowser.open("file://{}".format(full_filename))

# This table is a summary of all reviews across all movies.
# Blank spaces are movies that have not been rated by the user

# Looking through the table i can see that many of my spaces are blank
# I only have a relatively small amount of good data to work from. 

# This is called a sparse dataset. 
# Sparse data sets are normal for recommendation systems. 
# Most users will only review a small number of products so there will always be-
# a lot of blank data.