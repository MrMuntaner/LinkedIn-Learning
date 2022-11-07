import numpy as np
import pandas as pd
import matrix_factorization_utilities

# The data has been split into 2 different files.
# Training has 70% of the reviews and Test has 30%
# Load user ratings
raw_training_dataset_df = pd.read_csv('data/movie_ratings_data_set_training.csv')
raw_testing_dataset_df = pd.read_csv('data/movie_ratings_data_set_testing.csv')

# Creating separate ratings matrices for the training data and the test data 
ratings_training_df = pd.pivot_table(raw_training_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)
ratings_testing_df = pd.pivot_table(raw_testing_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)

# Apply matrix factorization to find the latent features on only the training data.
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_training_df.to_numpy(),
                                                                    num_features=11,
                                                                    regularization_amount=1.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Now i can calculate the error rates using the RMSE function and matrix factorization utilities 
# Measure RMSE
# To us the function i just need to pass in the 2 arrays that i want to compare
# First, i will compare the training data to the predicted ratings 
rmse_training = matrix_factorization_utilities.RMSE(ratings_training_df.to_numpy(),
                                                    predicted_ratings)

# Next, i do the same for the testing
rmse_testing = matrix_factorization_utilities.RMSE(ratings_testing_df.to_numpy(),
                                                    predicted_ratings)

print("Training RMSE: {}".format(rmse_training))
print("Testing RMSE: {}".format(rmse_testing))

# I got a training RMSE of Training RMSE: 0.24952589761404576
# and a Testing RMSE: 1.2096526122445328

# The low training RMSE shows that my basic algorithm is working.

# The testing RMSE is the more important number because it tells me how good my predictions are.
# A score of 1.2 means that my systems a bit more than one star off on average when predicting the users ratings

# One thing i can do to try to improve the system is to adjust the regularization amount parameter.
# However, this is a trade-off, more regularization will raise the training score, but it may lower the testing score.

# One limitation that i have in this example is that i only have a few hundred movie reviews to work with,-
# the best thing i could do to improve accuracy in this case is to get more user reviews.

# More movie reviews will give my system more information to work with so it can do a better job of making recommendations.