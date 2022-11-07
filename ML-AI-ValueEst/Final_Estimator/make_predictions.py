# from sklearn.externals 
import joblib

# notice that there are only a few lines of code even though this file contains all-
# the logic needed to estimate the value of a house.

# thats because all the logic was created by the ML algorithm and saved in my model.
# i only need to load the model, pass in house data, and run it.


# Load the gradient boosting model i trained previously
model = joblib.load('data/trained_house_classifier_model.pkl')

# For the house i want to value, i need to provide the features in the exact same
# arrangement as my training data set.

# this also means that the data here needs to reflect any feature engineering changes i made.
# so the fields i removed and the fields i added with one hot encoding are reflected here.

# i do this by creating the array below for an imaginary house built in 2006
house_to_value = [
    # House features
    1973,   # year_built 2006/1973
    1,      # stories
    4,      # num_bedrooms
    3,      # full_bathrooms
    0,      # half_bathrooms 
    2200,   # livable_sqft
    2350,   # total_sqft
    400,      # garage_sqft 0/400
    0,      # carport_sqft
    True,   # has_fireplace
    False,  # has_pool
    True,   # has_central_heating
    True,   # has_central_cooling
    
    # Garage type: Choose only one
    1,      # attached 0/1
    0,      # detached
    0,      # none 1/0
    
    # City: Choose only one
    1,      # Amystad 0/1
    0,      # Brownport 1/0
    0,      # Chadstad
    0,      # Clarkberg
    0,      # Coletown
    0,      # Davidfort
    0,      # Davidtown
    0,      # East Amychester
    0,      # East Janiceville
    0,      # East Justin
    0,      # East Lucas
    0,      # Fosterberg
    0,      # Hallfort
    0,      # Jeffreyhaven
    0,      # Jenniferberg
    0,      # Joshuafurt
    0,      # Julieberg
    0,      # Justinport
    0,      # Lake Carolyn
    0,      # Lake Christinaport
    0,      # Lake Dariusborough
    0,      # Lake Jack
    0,      # Lake Jennifer
    0,      # Leahview
    0,      # Lewishaven
    0,      # Martinezfort
    0,      # Morrisport
    0,      # New Michele
    0,      # New Robinton
    0,      # North Erinville
    0,      # Port Adamtown
    0,      # Port Andrealand
    0,      # Port Daniel
    0,      # Port Jonathanborough
    0,      # Richardport
    0,      # Rickytown
    0,      # Scottberg
    0,      # South Anthony
    0,      # South Stevenfurt
    0,      # Toddshire
    0,      # Wendybury
    0,      # West Ann
    0,      # West Brittanyview
    0,      # West Gerald
    0,      # West Gregoryview
    0,      # West Lydia
    0       # West Terrence
]

# create an array of the homes to value:
# scikit-learn assumes i want to predict the values for lots of houses at once, so it expects an array of houses to value.
# i just want to look at a single house, so it will be the only item in my array.
homes_to_value = [
    house_to_value
]

# making a prediction of the data:
# call model.predict and pass in the array of features

# Run the model and make a prediction for each house in the homes_to_value array
# the result of this is an array containing a price prediction for each house.
predicted_home_values = model.predict(homes_to_value)

# Since i am only predicting the price of one house, just look at the first prediction returned by looking at index zero
predicted_value = predicted_home_values[0]

print("This house has an estimated value of ${:,.2f}".format(predicted_value))

# for this house(2006)it predicts a price of $599,867
# Output: This house has an estimated value of $599,867.37

# but lets see what happens if i change some of the attributes.
# i will change the year from 2006 to 1973. re-run and see how it changes the value

# 1973 Output: This house has an estimated value of $456,051
# i can see that the value dropped to $456,051
# it looks like the age of the house matters a lot

# now what if i change some other attributes?
# what if i say that this house has an attached garage? 
# change it from attached and remove none, also add 400 for square feet of space.

# new Output: This house has an estimated value of $501,961.59
# now the value went up to $501,961

# what about the city? 
# changed city Output: This house has an estimated value of $377,916.41
# the value drops significantly to $377,916.
# it seems like location matters a lot to the value 