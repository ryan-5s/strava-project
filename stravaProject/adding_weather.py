import pandas as pd
import requests 
import time
from datetime import datetime, timedelta

runs_df = pd.read_csv('stravaProject/data/runs.csv')

#Convert to date-time object for easy formatting
runs_df.columns = runs_df.columns.str.replace(' ', '_')
runs_df['Activity_Date'] = pd.to_datetime(runs_df['Activity_Date'], format='mixed')
#Generate lists for adding temps and humidities
temps = []
humidities = []

#Using the unified Historical Forecast endpoint - much more stable for single-day tracking
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

print(f'Starting dynamic weather extraction for {len(runs_df)} runs.')

#Loop through runs using .itertuples() to read in row and column format for more efficient reading
for i, row in enumerate(runs_df.itertuples()):
    #YYYY-MM-DD for API 
    date_str = row.Activity_Date.strftime('%Y-%m-%d')
    
    #Find the hour for the run
    run_hour = row.Activity_Date.hour

    #Coordinates
    lat = row.latitude
    lon = row.longitude

    #Skip if invalid coordinates
    if pd.isna(lat) or pd.isna(lon) or (lat == 0.0 and lon == 0.0):
        print(f'Row {i} not working: Missing GPS coordinates')
        temps.append(None)
        humidities.append(None)
        continue


    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date_str,
        'end_date': date_str,
        'hourly': 'temperature_2m,relative_humidity_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'auto'
    }

    try:
        #Make reuqest over internet
        response = requests.get(url, params=params, timeout=5)

        #Check HTTP status code first
        if response.status_code != 200:
            print(f'Row {i} skipped ({date_str}): Server returned status code {response.status_code}')
            temps.append(None)
            humidities.append(None)
            continue 

        data = response.json()

        #Accounts for invalid data
        if 'error' in data:
            print(f'Row {i} skipped ({date_str}): API range restriction.')
            temps.append(None)
            humidities.append(None)
            continue 
        
        #Find temp and humidity
        hourly_temps = data['hourly']['temperature_2m']
        hourly_humidities = data['hourly']['relative_humidity_2m']

        temp = hourly_temps[run_hour]
        humidity = hourly_humidities[run_hour]
        
        temps.append(temp)
        humidities.append(humidity)
        print(f'Row {i} success ({date_str}): {temp} degrees F, {humidity}% humidity')

    except requests.exceptions.Timeout:
        print(f'Row {i} timed out. Server is rate-limiting user. Pausing.')
        temps.append(None)
        humidities.append(None)
        time.sleep(2)

    except Exception as e:
        print(f'Error fetching data for row index {i} on date {date_str}: {e}')
        temps.append(None)
        humidities.append(None)
    
    #IMPORTANT: Pauses for 0.2 s to prevent being blocked for overloading website
    time.sleep(0.2)

runs_df['start_temp'] = temps
runs_df['start_humidity'] = humidities

runs_df = runs_df.dropna(subset=['start_temp'])

#Create new csv file
runs_df.to_csv('stravaProject/data/runs_with_weather.csv', index=False)