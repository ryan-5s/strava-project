import matplotlib.pyplot as plt
import pandas as pd
import gpxpy
import os

#Function to find the coordinates of the run based on gpx file extraction
def get_start_coords(filename):

    #if filename is na, return none
    if pd.isna(filename):
        return None, None
    
    #Gnerating file path
    base_dir = r"C:\Users\ryanf\OneDrive\CHEM LABS\CODING\pandasLearning\stravaProject\data"
    gpx_folder = os.path.join(base_dir, "activities_gpx")

    #put together file name
    filename = str(filename).strip()

    if "/" in filename:
        filename = filename.split("/")[-1]

    file_path = os.path.join(gpx_folder, filename)

    #extracting longitude and latitude from gpx file
    if file_path.endswith('.gpx') and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
                if gpx.tracks and gpx.tracks[0].segments and gpx.tracks[0].segments[0].points:
                    start_point = gpx.tracks[0].segments[0].points[0]
                    return start_point.latitude, start_point.longitude
        except Exception as e:
            print(f'Error parsing file {filename}: {e}')
            return None, None
        
    
    return None, None


strava_df = pd.read_csv('stravaProject/data/activities.csv')
#convert strava data to runs over 0.5 mi, removing junk data and bike rides
strava_df['distance_miles'] = strava_df['Distance'] * 0.621371 #conversion from km to miles
strava_df['pace_min_per_mile'] = (strava_df['Elapsed Time'] / 60) / strava_df['distance_miles']
runs_df = strava_df[(strava_df['Activity Type'] == 'Run') & (strava_df['distance_miles'] >= 0.5)].copy()

#Get coordinates and add to runs for runs.csv
coordinates = runs_df['Filename'].apply(get_start_coords) #NOTE: apply() function allows only th filename column to be run, preventing slow and inefficient data processing

latitudes = []
longitudes = []

for coord in coordinates:
    if coord and coord[0] is not None:
        latitudes.append(coord[0])
        longitudes.append(coord[1])
    else:
        latitudes.append(None)
        longitudes.append(None)
runs_df['latitude'] = latitudes
runs_df['longitude'] = longitudes

runs_df.to_csv('stravaProject/data/runs.csv', index=False)
