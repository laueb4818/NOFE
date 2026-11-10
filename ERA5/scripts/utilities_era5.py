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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))
#os.chdir(os.path.dirname(__file__))

from utilities import GraphLMDBWriter
from generator_utility import (
    Graph_Generator,
    track_time,
    _function_timings,
    device,
    clear_memory_cache
)


EARTH_RADIUS_M = 6_371_000  # Earth's mean radius in meters; used for global ECEF coordinate normalization


#%%
class ERA5_Graph_Generator(Graph_Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
    

    @track_time
    def load_data(self, bbox:list=None, vars:list=None, time:str=None, filename:str=None):
        file_path = os.path.join(self.directory, filename)
        self.meta = Data(source_file=file_path)
        date = filename[:10]
        #print(date)
        self.meta['date'] = date
        
        min_lon, min_lat, max_lon, max_lat = bbox
        # initiallize self.da, self.grid_latitudes, self.grid_longitudes
        #file_name = os.path.join(datafolder_path, date + '.nc')
        
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            lat_max_min = (max_lat, min_lat)
            lon_min_max = (min_lon, max_lon)
        
        ds = xr.open_dataset(file_path, engine="netcdf4")
        
        # Convert all longitudes to [-180, 180] format
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        
        # Sort by longitude to ensure correct ordering after coordinate transformation
        ds = ds.sortby('longitude')
        
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
        for c in range(arr.shape[0]):
            channel = arr[c]
            mean = np.nanmean(channel)
            std = np.nanstd(channel)
            if std == 0:
                std = 1
            arr[c] = (channel - mean) / (std + 1e-8)
        self.da = arr
        # DEPRECATED: Previous normalization averaged across channels (different units/magnitudes),
        # which destroys physical meaning. Each channel (temperature, pressure, etc.) must be 
        # normalized independently to preserve comparability.
        # arr = ds.values        
        # mean = np.nanmean(arr, axis=0)  # shape (N1, N2) - spatial mean!
        # std = np.nanstd(arr, axis=0)
        # std[std == 0] = 1
        # self.da = (arr - mean) / std
        
        self.grid_latitudes = np.linspace(bbox[1], bbox[3], self.da.shape[1])
        self.grid_longitudes = np.linspace(bbox[0], bbox[2], self.da.shape[2])
        
    def cartesian_projection(self, lat:np.ndarray=None, lon:np.ndarray=None):
        """
        Convert GPS coordinates (lat/lon) to ECEF coordinates (x, y, z)
        Vectorized over all points.
        """
        
                
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
        Global ECEF coordinate normalization.

        Normalizes cartesian (ECEF) coordinates to approximately [-1, 1] in each
        dimension by dividing by sqrt(3) * EARTH_RADIUS_M. The center of Earth
        (0, 0, 0) is used as the reference, providing a consistent global frame
        of reference across all samples.
        """
        return cartesian / (np.sqrt(3) * EARTH_RADIUS_M)

    # --- PREVIOUS IMPLEMENTATION (per-sample per-bbox normalization) ---
    # Kept for reference; can be reverted if needed.
    #
    # def norm_coords(self, cartesian):
    #     """
    #     We norm the 3-dim. cartesian coordinates such that the values are [-1, 1].
    #     The scale of the 3 dimensions w.r.t each other remains the same.
    #     This is accomplished by subtracting their individual center values, but dividing by a common denominator
    #
    #     requires subregion to be loaded 'self.load_subregion()'
    #
    #     returns np.ndarray of shape (N,3)
    #     """
    #     bound_1 = self.cartesian_projection(np.min(self.grid_latitudes), np.min(self.grid_longitudes))
    #     bound_2 = self.cartesian_projection(np.min(self.grid_latitudes), np.max(self.grid_longitudes))
    #     bound_3 = self.cartesian_projection(np.max(self.grid_latitudes), np.min(self.grid_longitudes))
    #     bound_4 = self.cartesian_projection(np.max(self.grid_latitudes), np.max(self.grid_longitudes))
    #
    #     bounds = np.vstack(([bound_1, bound_2, bound_3, bound_4]))
    #
    #     x_min, y_min, z_min = np.min(bounds, axis=0)
    #     x_max, y_max, z_max = np.max(bounds,axis=0)
    #
    #     x_center = (x_max + x_min) / 2
    #     y_center = (y_max + y_min) / 2
    #     z_center = (z_max + z_min) / 2
    #
    #     denominator = np.sqrt(3) * np.sqrt( (x_max-x_min)**2 + (y_max-y_min)**2 + (z_max-z_min)**2 )
    #
    #     normed_coords = cartesian
    #     normed_coords[:,0] = (normed_coords[:,0] - x_center) / denominator
    #     normed_coords[:,1] = (normed_coords[:,1] - y_center) / denominator
    #     normed_coords[:,2] = (normed_coords[:,2] - z_center) / denominator
    #
    #     return normed_coords
    

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
        
        raw_coords = self.cartesian_projection(lat=latitudes, lon=longitudes)
        meta = Data(gps=gps)
        
        normed_coords = self.norm_coords(raw_coords)
        
        return normed_values, normed_coords, meta


    def synthetic_graph(self, N:int, k:int, gen_feature_mode:str="channel_only"):
        """
        Create a synthetic output graph with regular grid positions and dummy features.
        Used for inference with arbitrary grid resolutions.
        
        :param N: number of nodes in the output graph
        :type N: int
        :param k: number of nearest neighbors for edge creation
        :type k: int
        :param gen_feature_mode: determines node features - 'channel_only', 'position_only', 'channel_and_position'
        :type gen_feature_mode: str
        """
        n_channels, n_lat, n_lon = self.da.shape
        
        aspect = n_lat / n_lon
        n_lat_new = int(round(np.sqrt(N * aspect)))
        n_lon_new = int(round(N / n_lat_new))
        
        lat_vals = np.linspace(self.grid_latitudes[0], self.grid_latitudes[-1], n_lat_new)
        lon_vals = np.linspace(self.grid_longitudes[0], self.grid_longitudes[-1], n_lon_new)
        
        latitudes = np.repeat(lat_vals, n_lon_new)
        longitudes = np.tile(lon_vals, n_lat_new)
        
        raw_coords = self.cartesian_projection(lat=latitudes, lon=longitudes)
        normed_coords = self.norm_coords(raw_coords)

        normed_coords = torch.as_tensor(normed_coords, device=device, dtype=torch.float32)
        dummy_features = torch.zeros(len(latitudes), n_channels, device=device, dtype=torch.float32)
        
        meta = Data(gps=np.column_stack((latitudes, longitudes)), synthetic=True)
        
        edge_indices, edge_dist, _ = self.get_edges(
            source=normed_coords.cpu().numpy(), 
            target=normed_coords.cpu().numpy(), 
            k=k, 
            perplexity=0, 
            mode="self_search"
        )
        
        edge_indices = torch.as_tensor(edge_indices, device=device, dtype=torch.long)
        edge_dist = torch.as_tensor(edge_dist, device=device, dtype=torch.float32)
        
        node_features = self.get_node_features(
            normed_coords=normed_coords, 
            normed_values=dummy_features, 
            mode=gen_feature_mode
        )
        
        edge_attributes = self.get_edge_attributes(
            source_coords=normed_coords,
            target_coords=normed_coords,
            edge_indices=edge_indices,
            edge_dist=edge_dist,
            k=k,
            include_feat_diff=False
        )
        
        return Data(
            x=node_features, 
            edge_index=edge_indices, 
            edge_attr=edge_attributes, 
            edge_dist=edge_dist, 
            input_affinities=None, 
            normed_coords=normed_coords, 
            normed_values=dummy_features, 
            meta=meta
        )


class ERA5_Graph_Generator_perSampleNorm(ERA5_Graph_Generator):
    """
    ERA5_Graph_Generator variant that uses per-sample per-bbox coordinate normalization
    instead of the global ECEF normalization.

    Kept as a separate class for easy comparison and reverting.
    See ERA5_Graph_Generator.norm_coords() for the global version.
    """

    def norm_coords(self, cartesian):
        """
        Per-sample per-bbox coordinate normalization.

        Normalizes cartesian coordinates to approximately [-1, 1] by centering
        on the bounding box of the loaded sample and scaling by its diagonal.

        requires subregion to be loaded via load_data()
        """
        bound_1 = self.cartesian_projection(np.min(self.grid_latitudes), np.min(self.grid_longitudes))
        bound_2 = self.cartesian_projection(np.min(self.grid_latitudes), np.max(self.grid_longitudes))
        bound_3 = self.cartesian_projection(np.max(self.grid_latitudes), np.min(self.grid_longitudes))
        bound_4 = self.cartesian_projection(np.max(self.grid_latitudes), np.max(self.grid_longitudes))

        bounds = np.vstack(([bound_1, bound_2, bound_3, bound_4]))

        x_min, y_min, z_min = np.min(bounds, axis=0)
        x_max, y_max, z_max = np.max(bounds, axis=0)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        denominator = np.sqrt(3) * np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)

        normed_coords = cartesian.copy()
        normed_coords[:, 0] = (normed_coords[:, 0] - x_center) / denominator
        normed_coords[:, 1] = (normed_coords[:, 1] - y_center) / denominator
        normed_coords[:, 2] = (normed_coords[:, 2] - z_center) / denominator

        return normed_coords
