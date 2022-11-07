import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib

# remeber that ML models are only as accurate as the data i train it with.
# since house prices are always changing, the model will quickly get out of date-
# as the market changes.

# when the underlying data changes, i need to re-train the model. 
# i will do this here with a more updated house data set. 

# when i trained the estimator with the original data set my test set error was 59,927 dollars.
# lets see what happens when i re-train the model with an updated data set.

# with the updated data set the test error is 63,373 dollars. this is higher than the original.
# thats an increase in error of about 4,000 dollars.

# Load the updated data set
df = pd.read_csv("data/ml_house_data_set_updated.csv")

# Remove the fields from the data set that i don't want to include in my model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price from the feature data
del features_df['sale_price']

# Create the X and y arrays
X = features_df.to_numpy()
y = df['sale_price'].to_numpy()

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)

# Save the trained model to a file so i can use it in other programs
joblib.dump(model, 'trained_house_classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

