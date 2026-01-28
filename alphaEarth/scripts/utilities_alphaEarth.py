#%%
import sys
import os
import torch
import rasterio
from rasterio.windows import from_bounds
import xarray as xr
import numpy as np
import torch
from pyproj import Transformer
import faiss
from torch_geometric.data import Data
import pickle
import lmdb
from torch_scatter import scatter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from dataset_utilities import (
    GraphLMDBWriter,
    GraphLMDBReader,
    
)

from generator_utility import (
    Graph_Generator,
    track_time,
    _function_timings
)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'using Device: {device}')

# %% define graph_generator
class AlphaEarth_Graph_Generator(Graph_Generator):
    """
    This class will load data from alphaEarth data and is equipped with all functionalities to generate graph samples
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        
        
        #self.config = sample_config.copy()
        # Dynamically create attributes for easy access
        #for key, value in sample_config.items():
        #    setattr(self, key, value)

    @track_time
    def load_data(self, bbox:list=None, filename:str=None):
        """
        Load a bounding box from a large GeoTIFF and return as xarray.DataArray.
        
        bbox = (min_lon, min_lat, max_lon, max_lat)
        self.da will have structure: (channels, latitude, longitude)
        """
        print(f'\nloading data from {filename} ...')
        
        with rasterio.open(os.path.join(self.directory, filename)) as src:
            if bbox is None:
                # Read the entire raster
                arr = src.read()  # shape: (bands, height, width)
                transform = src.transform
            
            else:
                min_lon, min_lat, max_lon, max_lat = bbox
                # compute pixel window inside file
                window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                
                # efficient read — only reads the requested pixels
                arr = src.read(window=window)  # shape: (bands, h, w)
                
                # geotransform for the subset
                transform = src.window_transform(window)
            crs = src.crs
            
        # convert to xarray for ML-ready operations
        arr = xr.DataArray(
            arr,
            dims=("band", "y", "x"),
            coords={"band": np.arange(arr.shape[0])},
            attrs={"transform": transform, "crs": crs}
        ).values
        
        # TODO drop NAs can be improved
        if np.isnan(arr).any():
            print(f'WARNING: {np.isnan(arr).sum()} NaN detected (set to 0)')
        arr[np.isnan(arr)] = 0
        
        # normalize
        eps = 1e-8
        mean = arr.mean(axis=(1,2), keepdims=True)
        std = arr.std(axis=(1,2), keepdims=True) + eps
        print("arr finite:", np.isfinite(arr).all())
        print("arr nan count:", np.isnan(arr).sum())
        print("arr inf count:", np.isinf(arr).sum())
        print("arr min/max:", np.nanmin(arr), np.nanmax(arr))
        self.da = (arr - mean) / std 
        self.grid_latitudes = np.linspace(bbox[1], bbox[3], self.da.shape[1])
        self.grid_longitudes = np.linspace(bbox[0], bbox[2], self.da.shape[2])
        print(f'total number of points in dataset: {self.da.shape[1]*self.da.shape[2]}')
        print(f'self.da.shape: {self.da.shape}')
        print(f'grid_latitudes.shape: {self.grid_latitudes.shape}')
        print(f'grid_longitudes.shape: {self.grid_longitudes.shape}')



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


    def norm_coords(self, cartesian):
        """
        We norm the 3-dim. cartesian coordinates such that the values are [-1, 1].
        The scale of the 3 dimensions w.r.t each other remains the same.
        This is accomplished by subtracting their individual center values, but dividing by a common denominator
        
        requires subregion to be loaded 'self.load_subregion()'
        
        returns np.ndarray of shape (N,3)
        """
        bound_1 = self.cartesian_projection(np.min(self.grid_latitudes), np.min(self.grid_longitudes))
        bound_2 = self.cartesian_projection(np.min(self.grid_latitudes), np.max(self.grid_longitudes))
        bound_3 = self.cartesian_projection(np.max(self.grid_latitudes), np.min(self.grid_longitudes))
        bound_4 = self.cartesian_projection(np.max(self.grid_latitudes), np.max(self.grid_longitudes))
        
        bounds = np.vstack(([bound_1, bound_2, bound_3, bound_4]))
        
        x_min, y_min, z_min = np.min(bounds, axis=0)
        x_max, y_max, z_max = np.max(bounds,axis=0)
        
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        
        denominator = np.sqrt(3) * np.sqrt( (x_max-x_min)**2 + (y_max-y_min)**2 + (z_max-z_min)**2 )
                
        normed_coords = cartesian
        normed_coords[:,0] = (normed_coords[:,0] - x_center) / denominator
        normed_coords[:,1] = (normed_coords[:,1] - y_center) / denominator
        normed_coords[:,2] = (normed_coords[:,2] - z_center) / denominator
        
        return normed_coords


    @track_time
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
        
        
        
        cartesian = self.cartesian_projection(lat=latitudes, lon=longitudes)
        normed_coords = self.norm_coords(cartesian=cartesian)
        meta = Data(gps=gps)
        
        return normed_values, normed_coords, meta




# %%
