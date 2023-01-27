import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame, read_file
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import movingpandas as mpd
from shapely.geometry import Point
from pyproj import CRS
from holoviews import opts, dim
#from keplergl import KeplerGl

import warnings
warnings.simplefilter("ignore")

input_file = "ais_202102.csv"
df = pd.read_csv(input_file,sep=';', usecols=['date_time_utc', 'mmsi', 'lat', 'lon', 'sog', 'nav_status'])
df = df[df.sog>0]
print('Number of records: {} million'.format(round(len(df)/1000000)))

#df.drop(columns=['TranscieverClass'], inplace=True)
df.rename(columns={'date_time_utc':'time', 'mmsi':'id', 'lat':'lat', 'lon':'lon', 'nav_status':'navstat'}, inplace=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df['navstat'] = df['navstat'].astype('category')
#df['shiptype'] = df['shiptype'].astype('category')




print(df.head())
print(df.tail())
df.plot(x_compat=True)
plt.show()

df['sog'].hist(bins=100, figsize=(15,3))
plt.show()

print(df['id'])

print(f"Original size: {len(df)} rows")
df = df[df.sog>0]
print(f"Reduced to {len(df)} rows after removing 0 speed records")
df['sog'].hist(bins=100, figsize=(15,3))
plt.show()
print("here")

df['t'] = pd.to_datetime(df.index, format='%Y/%m/%d %H:%M:%S')
traj_collection = mpd.TrajectoryCollection(df, 'mmsi', t='t', min_length=100)
print(f"Finished creating {len(traj_collection)} trajectories")