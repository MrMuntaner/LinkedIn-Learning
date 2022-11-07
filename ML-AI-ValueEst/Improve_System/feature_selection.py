import numpy as np
# from sklearn.externals import joblib
import joblib

# With a tree based ML algorithm like radiant boosting, i can actually look at the train model-
# and have it tell me how often each feature is used in determining the final price.

# These are the feature labels from our data set
feature_labels = np.array(['year_built', 'stories', 'num_bedrooms', 'full_bathrooms', 'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft', 'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating', 'has_central_cooling', 'garage_type_attached', 'garage_type_detached', 'garage_type_none', 'city_Amystad', 'city_Brownport', 'city_Chadstad', 'city_Clarkberg', 'city_Coletown', 'city_Davidfort', 'city_Davidtown', 'city_East Amychester', 'city_East Janiceville', 'city_East Justin', 'city_East Lucas', 'city_Fosterberg', 'city_Hallfort', 'city_Jeffreyhaven', 'city_Jenniferberg', 'city_Joshuafurt', 'city_Julieberg', 'city_Justinport', 'city_Lake Carolyn', 'city_Lake Christinaport', 'city_Lake Dariusborough', 'city_Lake Jack', 'city_Lake Jennifer', 'city_Leahview', 'city_Lewishaven', 'city_Martinezfort', 'city_Morrisport', 'city_New Michele', 'city_New Robinton', 'city_North Erinville', 'city_Port Adamtown', 'city_Port Andrealand', 'city_Port Daniel', 'city_Port Jonathanborough', 'city_Richardport', 'city_Rickytown', 'city_Scottberg', 'city_South Anthony', 'city_South Stevenfurt', 'city_Toddshire', 'city_Wendybury', 'city_West Ann', 'city_West Brittanyview', 'city_West Gerald', 'city_West Gregoryview', 'city_West Lydia', 'city_West Terrence'])

# Load the trained model created with train_model.py
model = joblib.load('data/trained_house_classifier_model.pkl')

# Create a numpy array based on the model's feature importances
# in scikit learn this will gave me an array containing the feature importance for each feature.
# the total of all feature importances will add up to one. 

# i can think of this as a percentage rating of how often the feature is used in determining-
# a houses value.
importance = model.feature_importances_

# Sort the feature labels based on the feature importance rankings from the model
# using numpys argsort function to give a list of array indexes pointing to each element-
# in the array in order
feauture_indexes_by_importance = importance.argsort()

# Print each feature label, from most important to least important (reverse order)
for index in feauture_indexes_by_importance:
    print("{} - {:.2f}%".format(feature_labels[index], (importance[index] * 100.0)))
  
# At the bottom i can see that the last few features are the most important in the-
# houses price.
# Output:
# city_Rickytown - 0.00%
# city_New Robinton - 0.00%
# city_New Michele - 0.00%
# city_Martinezfort - 0.00%
# city_Julieberg - 0.00%
# city_Davidtown - 0.00%
# city_Lake Jennifer - 0.00%
# city_South Stevenfurt - 0.01%
# city_East Justin - 0.01%
# city_West Terrence - 0.01%
# city_Fosterberg - 0.02%
# city_West Brittanyview - 0.02%
# city_Leahview - 0.02%
# city_Brownport - 0.03%
# city_Joshuafurt - 0.03%
# city_Toddshire - 0.04%
# city_Port Adamtown - 0.04%
# city_Wendybury - 0.04%
# city_East Janiceville - 0.04%
# city_Amystad - 0.05%
# city_Port Daniel - 0.05%
# city_West Lydia - 0.08%
# city_Clarkberg - 0.08%
# garage_type_detached - 0.10%
# city_Davidfort - 0.10%
# has_central_cooling - 0.13%
# city_Port Jonathanborough - 0.15%
# city_West Gerald - 0.15%
# city_Jenniferberg - 0.16%
# city_Morrisport - 0.22%
# city_Lewishaven - 0.25%
# city_North Erinville - 0.26%
# city_East Amychester - 0.26%
# city_Lake Carolyn - 0.27%
# has_central_heating - 0.30%
# city_Lake Dariusborough - 0.30%
# city_West Gregoryview - 0.32%
# city_Richardport - 0.33%
# city_East Lucas - 0.34%
# city_South Anthony - 0.40%
# city_Justinport - 0.53%
# half_bathrooms - 0.56%
# city_West Ann - 0.58%
# city_Hallfort - 0.67%
# city_Chadstad - 1.09%
# city_Port Andrealand - 1.36%
# stories - 1.39%
# city_Scottberg - 1.47%
# city_Lake Christinaport - 1.47%
# city_Lake Jack - 1.69%
# num_bedrooms - 2.27%
# garage_type_none - 2.69%
# has_fireplace - 2.87%
# carport_sqft - 3.21%
# full_bathrooms - 3.39%
# year_built - 4.28%
# city_Coletown - 4.55%
# garage_sqft - 4.78%
# city_Jeffreyhaven - 4.84%
# garage_type_attached - 5.10%
# has_pool - 5.11%
# total_sqft - 12.57%
# livable_sqft - 28.93%

# if i ever have a really big model with hundreds or thousands of features i can-
# use this approuch to determine which features to keep and which features might be-
# better to throw out the next time i re-train it. 