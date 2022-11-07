import pandas
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Load the data set
df = pandas.read_csv("data/ml_house_data_set.csv")

# Remove the fields from the data set that i don't want to include in my model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
features_df = pandas.get_dummies(df, columns=['garage_type', 'city'])
del features_df['sale_price']

X = features_df.to_numpy()
y = df['sale_price'].to_numpy()

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create the model
model = ensemble.GradientBoostingRegressor()

# Using Grid Search:
# grid search is where i list out a range of settings i want to try for each parameter-
# and i literally try them all. 

# I train and test the model for every combination of parameters.
# the combination of parameters that generates the best predictions are the set of-
# parameters i should use for my real model. 

# this is almost the same code as train_model. the difference is that i declare my model-
# without passing in any parameters. 

# Parameters i want to try
# the pram_grid has an array for each parameter.
# for each setting, i add the range of values that i want to try. 
# the ranges i have here are good values to try for most problems.

# a good strategy is to try a few values for each parameter, where it increases-
# or decreases by a significant amount, like 1.0 to 0.3 to 0.1
param_grid = {
    'n_estimators': [500, 1000, 3000],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['ls', 'lad', 'huber']
}

# Define the grid search i want to run. Run it with four cpus in parallel.
# this takes in the model object, the param_grid, and the number of CPUs i want to use-
# to run my grid search.
# (in the future if i want to run this on a computer that has more than 1 cpu, i can speed things up by using all of them.)
gs_cv = GridSearchCV(model, param_grid, n_jobs=4)

# Run the grid search - on only the training data!
# dont give it access to the test data set
gs_cv.fit(X_train, y_train)

# Print the parameters that gave me the best result!
print(gs_cv.best_params_)

# After running a .....long..... time, the output will be something like
# {'loss': 'huber', 'learning_rate': 0.1, 'min_samples_leaf': 9, 'n_estimators': 3000, 'max_features': 0.1, 'max_depth': 6}

# That is the combination that worked best.

# Find the error rate on the training set using the best parameters
mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set using the best parameters
mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)