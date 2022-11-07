import pandas as pd
import webbrowser
import os

# This code is almost identical to the view_data file. 
# the difference is that i am passing the index_col parameter because i want the movie_id-
# as the index

# Read the dataset into a data table using Pandas
data_table = pd.read_csv("data/movies.csv", index_col="movie_id")

# Create a web page view of the data for easy viewing
html = data_table.to_html()

# Save the html to a temporary file
with open("movie_list.html", "w") as f:
    f.write(html)

# Open the web page in our web browser
full_filename = os.path.abspath("movie_list.html")
webbrowser.open("file://{}".format(full_filename))

# This is a list of all the movies in my data set.
# Each movie has both an id and a title. It also has the genres but i wont be using that.

# When starting a new recommendation project, its a great idea to look at the data visualy like this-
# to make sure i understand exactly what data i have to work with.