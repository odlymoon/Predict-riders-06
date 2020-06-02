"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('Datasets/Train.csv')
riders = pd.read_csv('Datasets/Riders.csv')
train = train.merge(riders, how='left', on='Rider Id')

y_train = train[['Time from Pickup to Arrival']]
X_train = train[['Experience', 'Distance (KM)', 'Temperature', 'No_Of_Orders',
               'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../trained-models/sendy_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
