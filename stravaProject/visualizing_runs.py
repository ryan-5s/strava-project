import pandas as pd
import matplotlib.pyplot as plt

#Pace vs. Temperature Graph
runs_df = pd.read_csv('stravaProject/data/runs_with_weather.csv')
runs_df.plot(kind='scatter', x='start_temp', y='pace_min_per_mile')
plt.title('Pace vs. Temperature')
plt.xlabel('Temperature (degrees F)')
plt.ylabel('Pace (min/mile)')
plt.show()

#Pace vs. Temp vs. Humidty while accounting for length of the run
runs_df.plot(kind='scatter',
             x='start_temp',
             y='pace_min_per_mile',
             c='start_humidity',#color mapping
             cmap='YlOrRd',#color map scale
             s=runs_df['distance_miles'] * 25, #dot size based on distance of run
             alpha=0.6,
             edgecolors='black',
             linewidths=0.5,
             figsize=(11, 7))

plt.title('Pace vs. Heat, Humidity, & Distance')
plt.xlabel('Temperature (degrees F)')
plt.ylabel('Pace (Minutes per Mile)')
plt.grid(True, linestyle='--', alpha=0.4)

plt.show()
