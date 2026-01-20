#%%
import sys 
import os
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from dataset_utilities import (
    GraphLMDBWriter,
    GraphLMDBReader,
)
from generator_utility import Graph_Generator


#%%
class ERA5_Graph_Generator(Graph_Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
    
    def load_data(self, bbox:list=None, vars:list=None, time:str=None, filename:str=None):
        file_path = os.path.join(self.directory, filename)
        self.date = filename[:10]
        print(self.date)
        
        min_lon, min_lat, max_lon, max_lat = bbox
        # initiallize self.da, self.grid_latitudes, self.grid_longitudes
        #file_name = os.path.join(datafolder_path, date + '.nc')
        
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            lat_max_min = (max_lat, min_lat)
            lon_min_max = (min_lon, max_lon)
        
        ds = xr.open_dataset(file_path)
        
        # Convert all longitudes to [-180, 180] format
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        
        # Apply longitude range condition
        if lon_min_max is not None: 
            l0, l1 = lon_min_max
            if l0 < l1:
                ds = ds.where((ds.longitude >= l0) & (ds.longitude <= l1), drop=True)
            else:  # Handle wrapping across -180/180
                ds = ds.where((ds.longitude >= l0) | (ds.longitude <= l1), drop=True)
                
        # Latitude selection
        if lat_max_min is not None:
            lat0, lat1 = sorted(lat_max_min)
            ds = ds.where((ds.latitude >= lat0) & (ds.latitude <= lat1), drop=True)
        
        # Time selection
        if time is not None and 'time' in ds:
            ds = ds.sel(time=time)
        # Variable selection
        if vars is not None:
            ds = ds[vars]
            
        # Convert to DataFrame
        df = ds.to_dataframe().reset_index()
        df = df.drop(columns=[c for c in ['valid_time', 'pressure_level', 'number', 'expver', 'longitude', 'latitude'] if c in df.columns], errors='ignore')
        #df = df.rename(columns={"longitude": "lon", "latitude": "lat"})
        
        # Normalize (ignoring NA)
        df[vars] -= df[vars].mean()
        df[vars] /= df[vars].std().replace(0, 1)
        
        # TODO: How to deal with NA? 
        # If remove, sampling a regular grid is not reliable (maybe interpolate then?)
        # Alternative: add masking layer + set value to mean / interpolate / ...
        # Set NaN to mean for now 
        # THIS IS NOT A GOOD SOLUTION BUT ONLY TO AVOID ERRORS DURING CODE TESTING
        df[vars] = df[vars].fillna(0.0)
                
        self.da = np.array(df[vars])
        self.grid_latitudes = np.linspaace
        self.grid_longitudes = np.array(df['lon'])
# %%
bbox = [10., 50, 15, 60]
vars = ['u', 't']
Gen = ERA5_Graph_Generator("/Volumes/T7-storage/ERA5/data/")
Gen.load_data(bbox=bbox, vars=vars, filename="2018-01-01.nc")
print(Gen.grid_latitudes.shape)
print(Gen.grid_longitudes.shape)
print(Gen.da.shape)

#%%
min_lon, min_lat, max_lon, max_lat = bbox
# initiallize self.da, self.grid_latitudes, self.grid_longitudes
#file_name = os.path.join(datafolder_path, date + '.nc')

if bbox is not None:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_max_min = (max_lat, min_lat)
    lon_min_max = (min_lon, max_lon)

ds = xr.open_dataset(file_path)

# Convert all longitudes to [-180, 180] format
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

# Apply longitude range condition
if lon_min_max is not None: 
    l0, l1 = lon_min_max
    if l0 < l1:
        ds = ds.where((ds.longitude >= l0) & (ds.longitude <= l1), drop=True)
    else:  # Handle wrapping across -180/180
        ds = ds.where((ds.longitude >= l0) | (ds.longitude <= l1), drop=True)
        
# Latitude selection
if lat_max_min is not None:
    lat0, lat1 = sorted(lat_max_min)
    ds = ds.where((ds.latitude >= lat0) & (ds.latitude <= lat1), drop=True)

# Time selection
if time is not None and 'time' in ds:
    ds = ds.sel(time=time)
# Variable selection
if vars is not None:
    ds = ds[vars]
