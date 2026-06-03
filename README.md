# Pace Tracker
Data from Strava, specifically the date, time, distance, and pace, were taken to investigate the correlations between parameters of temperature, humidity, and distance, on the pace of the run. 

## Idea
Measurements of temperature, humidity, and distance are accessed on their impact on the pace of the run. A model based on the parameters, using machine learning, is generated and used to find the expected pace (minutes per mile) of the run under set conditions of temperature (Fahrenheit), humidity (relative humidity), and distance (miles). 

## Scripts and Process
### data_cleaning.py
- Convert distance in kilometers to distance in miles and add to data frame
- Calculate pace in minutes per mile and add to data frame
- Find coordinates on .gpx files provided
- Add coordinates to data frame in latitude and longitude
- Only keep runs greater than 0.5 mile for runs.csv

### adding_weather.py
- Pull data from weather API for temperature and humidity on the given run's date and start time, then add to data frame
- Make runs_with_weather.csv and upload data frame

### visualizing_runs.py
- Basic plots for comparison between pace and factors of temperature, humidity, and distance
<img width="1265" height="956" alt="image" src="https://github.com/user-attachments/assets/98d00d87-1e3b-40f9-98cc-081ae1560e92" />
<img width="2008" height="1351" alt="image" src="https://github.com/user-attachments/assets/88e94863-e312-4406-b086-dd730741408a" />

### running_analysis.py
- Create linear and polynomial models up to degree 5 to draw correlation between independent variables of temperature, humidity, and distance, and the outcome of pace
- Find root mean square error for every model and output in units of minutes and seconds
- Find R squared values for every model and output
- For the linear regression model, plot the different variables and pace
<img width="2347" height="1519" alt="image" src="https://github.com/user-attachments/assets/51511350-d720-4be7-adb1-8159718ff08e" />
- Show a heat map of the correlations between each parameter and pace to determine what is and isn't significant
<img width="2497" height="1478" alt="image" src="https://github.com/user-attachments/assets/09dce6bc-5ea1-4b36-981c-5cc89511ba5b" />

### only_distance.py
- After finding that distance is the significant factor, create linear and polynomial models up to degree 5
- Output the root mean square error (in minutes and seconds) and the R squared value for every model
- Find the best fit model based on error and upload to models folder using joblib

### predict_pace.py
- Take user input of distance in miles
- Load the model using joblib
- Use model to predict pace and output
