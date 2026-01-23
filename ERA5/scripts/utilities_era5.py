#%%
import sys 
import os
import xarray as xr
import numpy as np
import pandas as pd
import json
import torch
from pyproj import Transformer
import faiss
from torch_geometric.data import Data
from torch_scatter import scatter

sys.path.append("/Volumes/T7-storage/NOFE/src/")
os.chdir("/Volumes/T7-storage/NOFE/ERA5/scripts/")

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
        self.meta = Data(source_file=file_path)
        date = filename[:10]
        print(date)
        self.meta['date'] = date
        
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
        
        # select only the variables in your list
        ds = ds.to_array(dim='variable') 
        ds = ds.squeeze(dim=['valid_time', 'pressure_level'])
        
        arr = ds.values        
        mean = np.nanmean(arr, axis=0)  # shape (N1, N2)
        std = np.nanstd(arr, axis=0)
        std[std == 0] = 1
        self.da = (arr - mean) / std
        
        self.grid_latitudes = np.linspace(bbox[1], bbox[3], self.da.shape[1])
        self.grid_longitudes = np.linspace(bbox[0], bbox[2], self.da.shape[2])
    '''
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
        print(bbox)
        print(self.da.shape)
        self.grid_latitudes = np.linspace(bbox[1], bbox[3], self.da.shape[1])
        self.grid_longitudes = np.linspace(bbox[0], bbox[2], self.da.shape[2])
    '''
    
    def generate_normed_sample(self, N:int=None, method:str=None):
        """
        This method generates a sample from initially loaded data self.da
        
        :param self: Description
        :param N: number of points to sample
        :type N: int
        :param method: 
        'random': randomly samples N points
        'interpolate': not defined yet
        'regular_subset': based on number of points N, the method selects a subset of the data that matches a regular grid structure
        :type method: str
        """
        
        n_channels, n_lat, n_lon = self.da.shape
        '''
        if N==None:
            N=self.N
        if method==None:
            method=self.generator_method
        '''
        
        if N >= n_lat*n_lon:
            print(f"requested number of points larger then dataset\n-> return full dataset of {n_lat*n_lon} points.")
            N = n_lat*n_lon
        
        if method=="random":
            # Flatten indices, sample without replacement
            flat_indices = np.random.choice(n_lat * n_lon, size=N, replace=False)
            # Convert back to 2D coordinates
            indices = np.column_stack(np.unravel_index(flat_indices, (n_lat, n_lon)))
            # sort by first and then second column
            indices = indices[np.lexsort((indices[:, 1], indices[:, 0]))]
        
        elif method=="interpolate":
            pass
        
        elif method=="regular_subset":
            aspect = n_lat / n_lon
            n_lat_new = int(round(np.sqrt(N * aspect)))
            n_lon_new = int(round(N / n_lat_new))  
            
            lat_inds = np.arange(n_lat)
            lat_inds_subset = lat_inds[np.linspace(0, len(lat_inds) - 1, n_lat_new, dtype=int)]
            
            lon_inds = np.arange(n_lon)
            lon_inds_subset = lon_inds[np.linspace(0, len(lon_inds) - 1, n_lon_new, dtype=int)]
            
            indices1 = np.repeat(lat_inds_subset, n_lon_new) # repeats each component 'n_lon_new' times
            indices2 = np.tile(lon_inds_subset, n_lat_new) # repeats full array 'n_lat_new' times 
            
            indices = np.vstack([indices1,indices2]).T
        
        else:
            print(f'"{method}" is an invalid sample method!')
        
        #self.normed_values = torch.tensor(self.da[:, indices[:,0], indices[:,1]].T)
        # more efficient. previously: normed_values = torch.tensor(self.da[:, indices[:,0], indices[:,1]].T)
        normed_values = torch.as_tensor(self.da[:, indices[:,0], indices[:,1]].T.copy())
        latitudes = self.grid_latitudes[indices[:,0]]
        longitudes = self.grid_longitudes[indices[:,1]]
        #self.gps = np.column_stack((latitudes, longitudes))
        gps = np.column_stack((latitudes, longitudes))
        
        return normed_values, gps
    
    
    def cartesian_projection(self, lat:np.ndarray=None, lon:np.ndarray=None):
        """
        Convert GPS coordinates (lat/lon) to ECEF coordinates (x, y, z)
        Vectorized over all points.
        """
        
        '''
        mode = "class_external"
        if lat == None:
            lat = self.gps[:, 0]
            mode = "class_internal"
        elif type(lat)==float:
            lat = np.array(lat)
        if lon == None:
            lon = self.gps[:, 1]
        elif type(lon)==float:
            lon = np.array(lon)
        '''
        
        altitude = 5500
        # Convert to arrays if floats
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        
        # Create transformer once
        trans = Transformer.from_crs(4979, 4978, always_xy=True)
        
        # Vectorized transform
        x, y, z = trans.transform(lon, lat, np.full_like(lat, altitude))
        
        # Stack into (N, 3)
        xyz = np.stack([x, y, z], axis=-1)
        
        # normalize
        #xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        
        cartesian=xyz
        return cartesian

# %%
with open('../dataset_configs/regular/example.json', 'r') as f:
    config = json.load(f)
bbox=[config['lon_min'], config['lat_min'], config['lon_max'], config['lat_max']]

#%%
G = ERA5_Graph_Generator(config['data_folder_path'])
G.load_data(bbox=bbox, vars=config['vars'], filename='2018-01-01.nc')

# %%
graph = G.sample_graph(N=100, k = 10, perplexity=10)
# %%

ds = xr.open_dataset(config['data_folder_path']+'/2018-01-01.nc')
# %%
print(ds.dims)

print(ds.shape)

# %%
da = ds[config['vars']]          # select only the variables in your list
da = da.to_array(dim='variable') 
da = da.squeeze(dim=['valid_time', 'pressure_level'])
data_array = da.values
data_array.shape
# %%
