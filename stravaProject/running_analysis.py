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

#Converts the minutes with decimal from the model prediction to a string in the format minutes:seconds
def convert_to_min_and_sec(raw_time):
    total_sec = raw_time * 60
    min = int(total_sec / 60)
    sec = round(total_sec % 60)
    return f'{min}:{sec:02d}'

#Prevents every single line from being executed when calling method in other file
if __name__ == '__main__':
    runs_df = pd.read_csv('stravaProject/data/runs_with_weather.csv')

    #Define the three independent variables
    x = runs_df[['start_temp', 'start_humidity', 'distance_miles']]

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
    print(f'MODEL WITH PARAMETERS OF TEMPERATURE, HUMIDITY, AND DISTANCE')
    print(f'The error for the linear regression model: {convert_to_min_and_sec(linear_error)}\nR²: {r2_linear}')

    #Plotting the linear regression model
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    #Averages accross three independent variables
    mean_temp = x['start_temp'].mean()
    mean_humidity = x['start_humidity'].mean()
    mean_dist = x['distance_miles'].mean()

    #Temperature subset - upper left 
    temp_range = np.linspace(x['start_temp'].min(), x['start_temp'].max(), 100)
    plot_temp = pd.DataFrame({'start_temp': temp_range, 'start_humidity':[mean_humidity] * 100, 'distance_miles':[mean_dist] * 100})

    axes[0, 0].scatter(x_train['start_temp'], y_train, label='Training Data')
    axes[0, 0].scatter(x_test['start_temp'], y_test, label = 'Testing Data')
    axes[0, 0].plot(temp_range, model.predict(plot_temp), label='Fit')
    axes[0, 0].set_title('Pace vs. Temperature')
    axes[0, 0].set_xlabel('Temperature (°F)')
    axes[0, 0].set_ylabel('Pace (min/mile)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    #Humidity subset - upper right 
    humidity_range = np.linspace(x['start_humidity'].min(), x['start_humidity'].max(), 100)
    plot_humidity = pd.DataFrame({'start_temp': [mean_temp] * 100, 'start_humidity': humidity_range, 'distance_miles':[mean_dist] * 100})

    axes[0, 1].scatter(x_train['start_humidity'], y_train)
    axes[0, 1].scatter(x_test['start_humidity'], y_test)
    axes[0, 1].plot(humidity_range, model.predict(plot_humidity))
    axes[0, 1].set_title('Pace vs. Humidity')
    axes[0, 1].set_xlabel('Humidity (%)')
    axes[0, 1].set_ylabel('Pace (min/mile)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    #Distance subset - bottom left 
    distance_range = np.linspace(x['distance_miles'].min(), x['distance_miles'].max(), 100)
    plot_distance = pd.DataFrame({'start_temp': [mean_temp] * 100, 'start_humidity': [mean_humidity] * 100, 'distance_miles':distance_range})

    axes[1, 0].scatter(x_train['distance_miles'], y_train)
    axes[1, 0].scatter(x_test['distance_miles'], y_test)
    axes[1, 0].plot(distance_range, model.predict(plot_distance))
    axes[1, 0].set_title('Pace vs. Distance')
    axes[1, 0].set_xlabel('Distance (miles)')
    axes[1, 0].set_ylabel('Pace (min/mile)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    #True pace vs. expected pace - bottom right
    axes[1, 1].scatter(y_test, y_pred_test, label='Test Runs')

    reference_line = np.linspace(min(y_test), max(y_test), 100)
    axes[1, 1].plot(reference_line, reference_line, label='Perfect Reference Line')

    axes[1, 1].set_title('Actual vs. Expected Pace')
    axes[1, 1].set_xlabel('Actual Pace (min/mile)')
    axes[1, 1].set_ylabel('Predicted Pace (min/mile)')
    axes[1, 1].grid(True, linestyle='--', alpha=.05)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

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
        joblib.dump(model_info, 'stravaProject/models/best_fit_model.pkl')
    else:
        model_info = {
            'model_type': 'linear',
            'transformer': None, 
            'model': lin_reg
        }
        joblib.dump(model_info, 'stravaProject/models/best_fit_model.pkl')

    # Select the features to test
    features = ['start_temp', 'start_humidity', 'distance_miles', 'pace_min_per_mile']
    corr_matrix = runs_df[features].corr()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.show()