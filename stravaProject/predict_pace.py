import pandas as pd
import joblib
from running_analysis import convert_to_min_and_sec

#Load regression model and unpack
model_info = joblib.load('stravaProject/models/best_fit_model.pkl')
model = model_info['model']
transformer = model_info['transformer']
model_type = model_info['model_type']

#Get user input for run
print(f'Use this tool to calculate your expected pace for your upcoming run!')
user_distance_string = input(f'Enter distance (miles): ')

#Convert to float
user_distance = float(user_distance_string)

#Enter into dataframe for modeling use
user_run = pd.DataFrame([{
    'distance_miles': user_distance
}])

#Predict pace and give to user
predicted_pace_raw = 0.0

if transformer is not None:
    user_run_transformed = transformer.transform(user_run)
    predicted_pace_raw = model.predict(user_run_transformed)[0]
else:
    predicted_pace_raw = model.predict(user_run)[0]

predicted_pace_string = convert_to_min_and_sec(predicted_pace_raw)
print(f'Your predicted pace is: {predicted_pace_string} min/mile')