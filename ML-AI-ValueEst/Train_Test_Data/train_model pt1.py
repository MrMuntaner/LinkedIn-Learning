import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib

# Preparing the desired features

# Load the data set
df = pd.read_csv("data/ml_house_data_set.csv")

# Remove the fields from the data set that I don't want to include in my model.
# there are 4 address fields that i want to remove from my dataset because they are-
# not useful to me.
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price from the feature data.
# i will delete the sale price column from my features because i dont want to-
# let the ML model see the sale price in the input data.
del features_df['sale_price']

# Create the X and y arrays:
# X is the standard name for the input features
# y is the standard name for the expected output to predict

# the X array will be the contents of my features dataframe. the only difference is that-
# i will call the as_matrix function to make sure the data is a numpy matrix data type-
# and not a pandas dataframe 
X = features_df.to_numpy()

# the y array will be the sales price column from my original dataset.
# i will also use the as_matrix function to convert it to a numpy matrix data type
y = df['sale_price'].to_numpy()

# Split the dataset in a training set (70%) and the test set (30%)
# when training a ML model i always need to do 2 things with my dataset

# first, shuffle the data so its in a random order 
# second, split the data into a training data set and a test data set

# the test_size parameter tells the model i want to keep 70% of the data for-
# training, and pull out 30% of the data for testing. a 70/30 split is pretty typical.

# splitting the data allows me to keep the test data hidden from the ML system-
# until i am ready to verify its accuracy.

# by using data the model has not seen before, it proves that the model actually-
# learned general rules for predicting house prices and it didnt just memorize-
# the answers for the specific houses it had seen before.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Creating and Training my ML model
# I will use the scikit-learn gradient boosting regressor. this will be one line of code!

# i also need to set the hyper-parameters that control how the gradient boosting regressor-
# model will run. n_estimators determine how many decision trees to build.

# learning rate controls how much each additional decision tree influences the overall prediction.
# lower rates usually lead to higher accuracy, but only works if i have n_estimators set to a high value.

# max_depth controls how many layers deep each individual decision tree can be.

# min_samples_leaf controls how many times a value must appear in my training set for a decision tree-
# to make a decision based on it.
# with this i am saying that at least 9 houses must exhibit the same characteristic before i consider-
# it meaningful enough to build a decision tree around it.

# this helps prevent single outliers from influencing the model too much.

# max_features is the percentage of features in my model that i randomly choose to consider each time-
# i create a branch in my decision tree.

# loss controls how scikit-learn calculates the error rate or cost as it learns.
# huber function does a good job while not being too influenced by outliers in the dataset.
# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber'
)

# next i tell the model to train using my training data set by calling scikit-learn's fit function on the model.
model.fit(X_train, y_train)

# a great feature of scikit-learn is that it uses the same interface for all supervised learning models it provides.

# Save the trained model to a file so we can use it in other programs
# this will let me use my train model again from a different program so that i can use it to make predictions for new houses. 
# joblib.dump(model, 'trained_house_classifier_model.pkl')


# After training a ML model, the next step is to measure how well the model performs.
# to check the accuracy of my models predictions, i will use a measurement called mean absolute error

# Mean absolute error looks at every prediction my model makes and gives me an average of how wrong it was-
# across all the predictions. 

# Find the error rate on the training set
# this will generate a prediction using my trained model for each entry in my training data set.
# scikit-learn will compare the predictions to the expected answers and tell me how close i am.
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

# for the training set the mean absolute error is 47,549 dollars.
# that means my model was able to predict the value of every house in my training data set to-
# within 47,000 dollars of the real price.

# considering the wide range of houses in my model, thats pretty good.

# for the test set the mean absolute error was a bit higher at 59,927 dollars.
# this tells me that my model still works for houses it has never seen before-
# but not quite as well as for the training houses.

# the difference in these two numbers tells me a lot about how well the model is working.