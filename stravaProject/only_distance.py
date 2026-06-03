import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns 
from running_analysis import convert_to_min_and_sec

runs_df = pd.read_csv('stravaProject/data/runs_with_weather.csv')
# Select the features you want to test
features = ['start_temp', 'start_humidity', 'distance_miles', 'pace_min_per_mile']
corr_matrix = runs_df[features].corr()

#Define the only independent variable of distance
x = runs_df[['distance_miles']]

#Define dependent variable (pace)
y = runs_df['pace_min_per_mile']

#Randomly split data for modeling and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Define and train model
model = LinearRegression(fit_intercept=True)
lin_reg = model.fit(x_train, y_train)
   
#Get model parameters
m_linear = lin_reg.coef_
c_linear = lin_reg.intercept_

#Test model
y_pred_test = lin_reg.predict(x_test)
linear_error = math.sqrt(mean_squared_error(y_test, y_pred_test))
r2_linear = r2_score(y_test, y_pred_test)
print(f'MODEL WITH PARAMETERS OF TEMPERATURE')
print(f'The error for the linear regression model: {convert_to_min_and_sec(linear_error)}\nR²: {r2_linear}')

#Polynomial Fit
#First use polynomial model of degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_reg = LinearRegression(fit_intercept=True)
poly_reg.fit(x_train_poly, y_train)

#Find error for degree 2
y_pred_poly_test = poly_reg.predict(x_test_poly)
poly_error = math.sqrt(mean_squared_error(y_test, y_pred_poly_test))
r2_poly2 = r2_score(y_test, y_pred_poly_test)

print(f'The error for Polynomial model of degree 2 is: {convert_to_min_and_sec(poly_error)}\nR²: {r2_poly2}')

#Run for loop to find which degree polynomial is the most accurate 
final_degree = 2
final_poly_error = poly_error

for d in range(3, 6):
    #Using temporary (temp) variables as placeholders just in case it is a more accurate model than second degree
    temp_poly = PolynomialFeatures(degree=d, include_bias=False)
    temp_x_train_poly = temp_poly.fit_transform(x_train)
    temp_x_test_poly = temp_poly.transform(x_test)

    temp_poly_reg = LinearRegression(fit_intercept=True)
    temp_poly_reg.fit(temp_x_train_poly, y_train)

    #Find error for degree d
    temp_y_pred_poly_test = temp_poly_reg.predict(temp_x_test_poly)
    temp_poly_error = math.sqrt(mean_squared_error(y_test, temp_y_pred_poly_test))
    temp_poly_r2 = r2_score(y_test, temp_y_pred_poly_test)
    print(f'Polynomial model of degree {d} error: {convert_to_min_and_sec(temp_poly_error)}\nR²: {temp_poly_r2}')

    #Replaces final degree and updates model if error is lower
    if temp_poly_error < final_poly_error:
        final_poly_error = temp_poly_error
        final_degree = d
        poly = temp_poly
        x_train_poly = temp_x_train_poly
        x_test_poly = temp_x_test_poly
        poly_reg = LinearRegression(fit_intercept=True)
        poly_reg.fit(x_train_poly, y_train)

print(f'Final degree: {final_degree}')

#Finds which model has less error, and then saves the features of that model
if final_poly_error <= linear_error:
    model_info = {
           'model_type': 'polynomial',
           'transformer': poly,
           'model': poly_reg
        }
        
else:
    model_info = {
        'model_type': 'linear',
        'transformer': None, 
        'model': lin_reg
        }

#Finds which model has less error, and then saves the features of that model
if final_poly_error <= linear_error:
    model_info = {
        'model_type': 'polynomial',
        'transformer': poly,
        'model': poly_reg
    }
    joblib.dump(model_info, 'stravaProject/models/best_fit_model.pkl')
else:
    model_info = {
        'model_type': 'linear',
        'transformer': None, 
        'model': lin_reg
    }
    joblib.dump(model_info, 'stravaProject/models/best_fit_model.pkl')
