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
<img width="1938" height="1326" alt="image" src="https://github.com/user-attachments/assets/3dc1ee88-3544-4bab-95aa-ab62455c118a" />
### 
