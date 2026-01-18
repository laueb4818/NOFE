# %%
"""
Subgraph Dataset Creation Script for AlphaEarth Satellite Data
===================================================================

Creates train and test LMDB datasets with two-subgraph structure from AlphaEarth GeoTIFF data.
This is an alternative to create_dataset.py with a different graph topology.

Graph structure:
- Input graph: Random subset of points with k-NN connectivity
- Output graph: All points with k-NN connectivity  
- Cross edges: Connections from output nodes to nearest input nodes

Usage:
    python create_subgraph_dataset.py <config_name> [--no-test] [--data_path PATH]
    
    config_name: Name of config file in dataset_configs_subgraph/ (without .json extension)
    --no-test: Optional flag to skip test dataset creation
    --data_path: Path to AlphaEarth data directory (default: ../data/)

Examples:
    python create_subgraph_dataset.py subgraph_config
    # Uses default: ../data/
"""

# %% loading alpha Earth data to memory

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

import sys
import os
import argparse
from pathlib import Path
import torch

# %%

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/alphaEarth/scripts/" )
else:
    device = torch.device("cpu")
print(device)
sys.path.append("../../src/dataset_utilities.py")


from utilities_alphaearth import (
    setup_device,
    load_config,
    validate_subgraph_config,
    split_train_test,
    create_subgraph_lmdb_dataset,
    set_seed,
)

# %%

def print_subgraph_summary(config_name, train_years, test_years, config):
    """Print summary of subgraph dataset configuration."""
    total_years = len(train_years) + len(test_years)
    
    # Determine if using manual or automatic split
    has_manual_split = 'train_years' in config and 'test_years' in config
    
    if has_manual_split:
        train_samples = config['train_samples_per_year']
        test_samples = config['test_samples_per_year']
    else:
        train_samples = config.get('samples_per_year')
        test_samples = config.get('samples_per_year')
    
    print("=" * 60)
    print(f"Subgraph Dataset Configuration: {config_name}")
    print("=" * 60)
    print(f"Total years:          {total_years}")
    print(f"  Training years:     {len(train_years)}")
    print(f"  Test years:         {len(test_years)}")
    if has_manual_split:
        print(f"Samples per year:     train={train_samples}, test={test_samples}")
        print(f"  Total train samples: {len(train_years) * train_samples}")
        print(f"  Total test samples:  {len(test_years) * test_samples}")
    else:
        print(f"Samples per year:     {train_samples}")
        print(f"  Total train samples: {len(train_years) * train_samples}")
        print(f"  Total test samples:  {len(test_years) * test_samples}")
    print(f"\nGraph parameters:")
    print(f"  k_in (input):       {config['k_in']}")
    print(f"  k_out (output):     {config['k_out']}")
    print(f"  k_cross (cross):    {config['k_cross']}")
    print(f"  Perplexity:         {config['perplexity']}")
    print(f"  Input nodes:        {config['n_points_input_min']}-{config['n_points_input_max']} (random)")
    
    if 'lat_min' in config and 'lat_max' in config and 'lon_min' in config and 'lon_max' in config:
        print(f"\nRegion bounds:")
        print(f"  Latitude:           [{config['lat_min']}, {config['lat_max']}]")
        print(f"  Longitude:          [{config['lon_min']}, {config['lon_max']}]")
    
    print(f"\nYears: {', '.join(map(str, train_years + test_years))}")
    print("=" * 60 + "\n")




# %% Testing around

graph_config={
    "data_folder_path": "/Volumes/T7-storage/ERA5/data/",
    "k_in": 5,
    "k_out": 5,
    "k_cross": 3,
    "perplexity": 15,
    "n_points_input_min": 1000,
    "n_points_input_max": 2000,
    "n_points_lat_out": 100,
    "n_points_lon_out": 100,
    "train_samples_per_year": 100,
    "test_samples_per_year": 100,
    "lat_min": 59.66,
    "lat_max": 59.73,
    "lon_min": 10.79,
    "lon_max": 10.86,
    "train_years": [2018, 2019, 2020],
    "test_years": [2021],
    "cache_region_data": True
    }

dataset_config = {
    "data_folder_path": "/Volumes/T7-storage/ERA5/data/",
    "k_in": 5,
    "k_out": 5,
    "k_cross": 3,
    "perplexity": 15,
    "n_points_input_min": 1000,
    "n_points_input_max": 2000,
    "n_points_lat_out": 100,
    "n_points_lon_out": 100,
    "train_samples_per_year": 100,
    "test_samples_per_year": 100,
    "lat_min": 59.66,
    "lat_max": 59.73,
    "lon_min": 10.79,
    "lon_max": 10.86,
    "train_years": [2018, 2019, 2020],
    "test_years": [2021],
    "cache_region_data": True
    }

sample_config = {
    "data_folder_path": "/Volumes/T7-storage/alphaEarth/data",
    "filename": "sat_emb_2018.tif",
    "bbox": (10.79, 59.66, 10.86, 59.73),
    "k": 5
}

# %% define graph_generator
class AlphaEarth_Graph_Generator:
    """
    This class will load data from alphaEarth data and is equipped with all functionalities to generate graph samples
    """
    def __init__(self, sample_config: dict):
        self.config = sample_config.copy()
        
        # Dynamically create attributes for easy access
        for key, value in sample_config.items():
            setattr(self, key, value)

    def load_subregion(self):
        """
        Load a bounding box from a large GeoTIFF and return as xarray.DataArray.
        
        bbox = (min_lon, min_lat, max_lon, max_lat)
        self.da will have structure: (channels, latitude, longitude)
        """
        
        with rasterio.open(os.path.join(self.data_folder_path, self.filename)) as src:
            if self.bbox is None:
                # Read the entire raster
                arr = src.read()  # shape: (bands, height, width)
                transform = src.transform
            
            else:
                min_lon, min_lat, max_lon, max_lat = self.bbox
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
        mean = arr.mean(axis=(1,2), keepdims=True)
        std = arr.std(axis=(1,2), keepdims=True)
        self.da = (arr - mean) / std
        self.grid_latitudes = np.linspace(self.bbox[1], self.bbox[3], self.da.shape[1])
        self.grid_longitudes = np.linspace(self.bbox[0], self.bbox[2], self.da.shape[2])
        print(f'total number of points in dataset: {self.da.shape[1]*self.da.shape[2]}')
        print(f'self.da.shape: {self.da.shape}')
        print(f'grid_latitudes.shape: {self.grid_latitudes.shape}')
        print(f'grid_longitudes.shape: {self.grid_longitudes.shape}')


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
        normed_values = torch.tensor(self.da[:, indices[:,0], indices[:,1]].T)
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


    def get_edges(self, source:np.ndarray=None, target:np.ndarray=None, k:int=None, mode:str="self_search"):
        """
        Searches for k nearest neighbors of 'target' in 'source'.
        
        :param source: dataset to find neighbors in
        :type source: np.ndarray
        :param target: dataset with points that neighbors are searched for 
        :type target: np.ndarray
        :param k: number of neighbors
        :type k: int
        :param cross_search: determines operations
        :type cross_search: bool
        """
        
        N, D = source.shape
        index = faiss.IndexFlatL2(D)  # L2 distance index (Euclidean)
        index.add(source)
        if mode == "cross_search":
            distances, indices = index.search(target, k) 
        elif mode == "self_search": 
            distances, indices = index.search(target, k+1) 
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            print("no valid mode was provided!")
        
        
        source_idx = indices.reshape(-1)
        target_idx = np.repeat(np.arange(target.shape[0]),k)
        edge_indices = torch.tensor(np.vstack([source_idx, target_idx]))
        
        
        edge_dist = distances.reshape(-1, 1)
        
        distance_table, neighbor_table = distances, indices # renaming to match other method
        
        return distance_table, neighbor_table, edge_indices, edge_dist


    def get_edge_attributes_new(self, source_coords:np.ndarray=None, 
                            target_coords:np.ndarray=None, 
                            edge_indices:torch.Tensor=None, 
                            edge_dist:np.ndarray=None,
                            distance_table:np.ndarray=None,
                            k:int=None,
                            include_feat_diff:bool=False):
        """
        Docstring for get_edge_attributes
        
        :param self: Description
        :param source_coords: normed_coords of source graph
        :type source_coords: np.ndarray
        :param target_coords: normed_coords of target graph
        :type target_coords: np.ndarray
        :param edge_indices: Description
        :type edge_indices: torch.Tensor
        :param edge_dist: Description
        :type edge_dist: np.ndarray
        :param distance_table: Description
        :type distance_table: np.ndarray
        :param k: Description
        :type k: int
        :param include_feat_diff: Description
        :type include_feat_diff: bool
        """
        
        # target node positions
        node_pos = target_coords[edge_indices[1,:]]
        
        max_neighbor_dists = np.repeat(distance_table[:,k-1], k)
        regularized_distances = edge_dist.squeeze()/max_neighbor_dists
        regularized_distances = regularized_distances[:, np.newaxis] 
        
        # normed displacement vector
        source_nodes = source_coords[edge_indices[0,:]]
        target_nodes = target_coords[edge_indices[1,:]]
        absolute_displacement = source_nodes - target_nodes
        epsilon = 1e-12
        normed_displacement = absolute_displacement / (np.linalg.norm(absolute_displacement, axis=1, keepdims=True) + epsilon)
        
        # NOTE optionally include date encoding, etc.
        if include_feat_diff:
            print('including feature difference not yet implemented. Continuing without')
        
        # bind it all together and make it a tensor
        edge_attributes = np.hstack([node_pos, normed_displacement, regularized_distances])
        edge_attributes = torch.tensor(edge_attributes)
        
        return edge_attributes


    '''
    def knn(self, X:np.ndarray=None, k:int=None):
        """
        Docstring for knn
        
        :param self: Description
        :param k: Description
        :type k: int
        :param X: Description
        """
        #if X==None:
        #    X = self.normed_coords
        #X = X.astype('float32')
        #if k == None:
        #    k=self.k
        
        N, D = X.shape
        index = faiss.IndexFlatL2(D)  # L2 distance index (Euclidean)
        index.add(X)
        
        # Search for k+1 neighbors to exclude the point itself (first neighbor is always self)
        distances, indices = index.search(X, k + 1)
        
        # Exclude self-match
        distance_table = distances[:, 1:]
        neighbor_table = indices[:, 1:]
        
        return distance_table, neighbor_table
    '''


    '''
    def build_edge_index(self, distance_table:np.ndarray=None, neighbor_table:np.ndarray=None):
        """
        Build PyTorch Geometric edge_index for incoming kNN graph.
        """
        N, k = neighbor_table.shape
        
        # Each query node connects to its neighbors
        src = np.repeat(np.arange(N), k)  # query nodes
        dst = neighbor_table.reshape(-1)  # neighbors
        # Stack into PyG format
        edge_indices = torch.tensor(
            np.vstack([src, dst]),
            dtype=torch.long
        )
        # distance attribute per edge
        edge_dist = distance_table.reshape(-1, 1)

        
        return edge_indices, edge_dist
    '''


    '''   
    # test replacement of above method:
    def build_cross_edge_index(
            distance_table: np.ndarray,
            neighbor_table: np.ndarray
        ):
        """
        Build edge_index for cross-kNN graph.
        
        Semantics:
        - A (reference set) = SOURCE nodes (send messages)
        - B (query set)     = TARGET nodes (receive messages)
        
        edge_index[0] -> indices into A  (sources)
        edge_index[1] -> indices into B  (targets)
        """
        
        N_B, k = neighbor_table.shape  # number of target nodes (B)
        
        # SOURCE nodes: k nearest neighbors in A for each B
        src = neighbor_table.reshape(-1)           # shape: (N_B * k,)
        
        # TARGET nodes: each B node repeated k times
        dst = np.repeat(np.arange(N_B), k)          # shape: (N_B * k,)
        
        edge_index = torch.tensor(
            np.vstack([src, dst]),
            dtype=torch.long
        )
        
        edge_dist = distance_table.reshape(-1, 1)
        
        return edge_index, edge_dist
    '''

    '''
    def get_cross_edge_attributes(self, 
                            source_normed_coords:np.ndarray=None, 
                            target_normed_coords:np.ndarray=None,
                            edge_indices:torch.Tensor=None, 
                            edge_dist:np.ndarray=None,
                            distance_table:np.ndarray=None,
                            k:int=None,
                            include_feat_diff:bool=False):
        
        # target node positions
        node_pos = target_normed_coords[edge_indices[1,:]]
        
        max_neighbor_dists = np.repeat(distance_table[:,k-1], k)
        regularized_distances = edge_dist.squeeze()/max_neighbor_dists
        regularized_distances = regularized_distances[:, np.newaxis] 
        
        # normed displacement vector
        source_nodes = source_normed_coords[edge_indices[0,:]]
        target_nodes = target_normed_coords[edge_indices[1,:]]
        absolute_displacement = source_nodes - target_nodes
        normed_displacement = absolute_displacement / np.linalg.norm(absolute_displacement, axis=1, keepdims=True)
        
        # NOTE optionally include date encoding, etc.
        if include_feat_diff:
            print('including feature difference not yet implemented. Continuing without')
        
        # bind it all together and make it a tensor
        edge_attributes = np.hstack([node_pos, normed_displacement, regularized_distances])
        edge_attributes = torch.tensor(edge_attributes)
        
        return edge_attributes
    '''


    '''
    def get_edge_attributes(self, normed_coords:np.ndarray=None, 
                            edge_indices:torch.Tensor=None, 
                            edge_dist:np.ndarray=None,
                            distance_table:np.ndarray=None,
                            k:int=None,
                            include_feat_diff:bool=False):
        
        # target node positions
        node_pos = normed_coords[edge_indices[1,:]]
        
        max_neighbor_dists = np.repeat(distance_table[:,k-1], k)
        regularized_distances = edge_dist.squeeze()/max_neighbor_dists
        regularized_distances = regularized_distances[:, np.newaxis] 
        
        # normed displacement vector
        source_nodes = normed_coords[edge_indices[0,:]]
        target_nodes = normed_coords[edge_indices[1,:]]
        absolute_displacement = source_nodes - target_nodes
        normed_displacement = absolute_displacement / np.linalg.norm(absolute_displacement, axis=1, keepdims=True)
        
        # NOTE optionally include date encoding, etc.
        if include_feat_diff:
            print('including feature difference not yet implemented. Continuing without')
        
        # bind it all together and make it a tensor
        edge_attributes = np.hstack([node_pos, normed_displacement, regularized_distances])
        edge_attributes = torch.tensor(edge_attributes)
        
        return edge_attributes
    '''

    def get_node_features(self, normed_coords:np.ndarray=None, normed_values:torch.Tensor=None, include_pos:bool=True):
        if include_pos:
            pos_tensor = torch.tensor(normed_coords)
            node_features = torch.cat([normed_values, pos_tensor], dim=1)
        else:
            node_features = normed_values
        return node_features


    def get_sigmas(self, distance_table:np.ndarray=None, perplexity:int=None):
        """
        Compute bandwidth (sigma) for Gaussian kernel based on perplexity.
        Similar to t-SNE perplexity tuning.
        
        Args:
            distances: np.array of shape (n_points, k) - distances to k neighbors
            perplexity: target perplexity value
        
        Returns:
            sigmas: np.array of shape (n_points,)
        """
        
        '''
        mode = "class_external"
        
        if distances==None:
            mode = "class_internal"
            distances = self.distance_table
        if perplexity==None:
            perplexity=self.perplexity
        '''
        
        target_entropy = np.log(perplexity)
        
        def compute_entropy(sigma, dists):
            """Compute entropy given sigma."""
            # Gaussian kernel
            p = np.exp(-dists**2 / (2 * sigma**2))
            p = p / (np.sum(p) + 1e-10)
            entropy = -np.sum(p * np.log(p + 1e-10))
            return entropy
        
        sigmas = []
        for i in range(distance_table.shape[0]):
            dists = distance_table[i]
            
            # Binary search for sigma
            sigma_min, sigma_max = 0.01, 1000.0
            sigma = 1.0
            
            for _ in range(50):  # max iterations
                entropy = compute_entropy(sigma, dists)
                
                if abs(entropy - target_entropy) < 1e-5:
                    break
                
                if entropy > target_entropy:
                    sigma_max = sigma
                    sigma = (sigma + sigma_min) / 2
                else:
                    sigma_min = sigma
                    sigma = (sigma + sigma_max) / 2
            
            sigmas.append(sigma)
        
        return np.array(sigmas)



    def get_affinities(self, normed_values:torch.Tensor=None, edge_indices:torch.Tensor=None, sigmas:np.ndarray=None):
        """
        Compute local affinity matrix using per-point sigma values.

        Args:
            var_tens (Tensor): [N, D] input features
            edges (LongTensor): [2, E] edge list, edges[:,0] = source, edges[:,1] = target
            sigmas (Tensor): [N] per-point sigma values for each source node

        Returns:
            Tensor: [N, k] affinity values (p_j|i) normalized per source node
        """
        '''
        mode = "class_external"
        if var_tens==None:
            mode = "class_internal"
            var_tens=self.normed_values
        if edges_indices==None:
            edges_indices=self.edge_indices
        if sigmas==None:
            self.get_sigmas()
            sigmas=self.sigmas
        '''
        
        eps = 1e-6
        
        # Get node embeddings for source and target
        x_i = normed_values[edge_indices[0, :]]  # [E, D]
        x_j = normed_values[edge_indices[1, :]]  # [E, D]
        sigma_i = sigmas[edge_indices[0, :]]  # [E]
        
        # Compute squared distances
        sqdist = torch.sum((x_i - x_j) ** 2, dim=1)
        
        # Compute affinities using Gaussian kernel with per-point sigma
        denom = 2 * sigma_i ** 2 + eps
        v = torch.exp(-sqdist / denom)  # [E]
        
        # Group v into rows of shape [N, k]
        N = normed_values.shape[0]
        k = v.numel() // N
        v = v.view(N, k)
        
        # Normalize to get conditional probabilities p_j|i
        p = v / (v.sum(dim=1, keepdim=True) + eps)  # [N, k]
        
        return sigmas, p #input affinities


    def sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random"):
        normed_values, gps = self.generate_normed_sample(N=N, method=sampling_method) # sample subset 
        cartesian = self.cartesian_projection(lat=gps[:, 0], lon=gps[:, 1]) # get 3 dim. cartesian coordinates for sample locations
        normed_coords = self.norm_coords(cartesian=cartesian) # norm coordinates based on subregion
        # old: distance_table, neighbor_table = self.knn(X=normed_coords, k=k) # get neighbors and distances as table
        # old: edge_indices, edge_dist = self.build_edge_index(distance_table=distance_table, neighbor_table=neighbor_table) # reshape to torch usable format in directed graph structure
        # old: edge_attributes = self.get_edge_attributes(normed_coords=normed_coords, 
        #                    edge_indices=edge_indices, 
        #                    edge_dist=edge_dist,
        #                    distance_table=distance_table,
        #                    k=k,
        #                    include_feat_diff=self.include_feat_diff)
        distance_table, neighbor_table, edge_indices, edge_dist = self.get_edges(source=normed_coords, target=normed_coords, k=k, mode="self_search")
        edge_attributes = self.get_edge_attributes_new(source_coords=normed_coords, 
                            target_coords=normed_coords, 
                            edge_indices=edge_indices, 
                            edge_dist=edge_dist,
                            distance_table=distance_table,
                            k=k,
                            include_feat_diff=False)
        node_features = self.get_node_features(normed_coords=normed_coords, normed_values=normed_values, include_pos=self.nodeFeat_include_pos)
        sigmas = self.get_sigmas(distance_table=distance_table, perplexity=perplexity)
        input_affinities = self.get_affinities(normed_values=normed_values, edge_indices=edge_indices, sigmas=sigmas)
        return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, input_affinities=input_affinities, sigmas=sigmas, gps=gps, normed_coords=normed_coords, source=os.path.join(self.data_folder_path, self.filename))


    #def cross_knn(self, A, B, k):
    #    # 1. Build index
    #    index = faiss.IndexFlatL2(A.shape[1])  # L2 distance
    #    index.add(A)  # add reference points
    #    
    #    # 2. Search
    #    D, I = index.search(B, k)
    #    distance_table = D
    #    neighbor_table = I
    #    
    #    return distance_table, neighbor_table


    def connect_graphs(self, in_graph:Data=None, out_graph:Data=None, k_cross:int=None):
        """
        This method creates a cross graph structure.
        the output is something like
        out_graph = sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random")
        
        returns list of Data objects
        !!! Careful when writing data as it is in form of a list !!!
        """
        cross_distance_table, cross_neighbor_table, cross_edge_indices, cross_edge_dist = self.get_edges(source=in_graph.normed_coords, 
                                                                                 target=out_graph.normed_coords, 
                                                                                 k=k_cross, mode="cross_search")
        cross_edge_attributes = self.get_edge_attributes_new(source_coords=in_graph.normed_coords, 
                            target_coords=out_graph.normed_coords, 
                            edge_indices=cross_edge_indices, 
                            edge_dist=cross_edge_dist,
                            distance_table=cross_distance_table,
                            k=k_cross,
                            include_feat_diff=False)
        # old: cross_distance_table, cross_neighbor_table = self.cross_knn(A=in_graph.normed_coords, B=out_graph.normed_coords, k=k_cross)
        # old: cross_edge_indices, cross_edge_dist = self.build_edge_index(distance_table=cross_distance_table, neighbor_table=cross_neighbor_table)
        # old: cross_edge_attributes = self.get_cross_edge_attributes(source_normed_coords=in_graph.normed_coords, 
        #                                                        target_normed_coords=out_graph.normed_coords,
        #                                                        edge_indices=cross_edge_indices,
        #                                                        edge_dist=cross_edge_dist,
        #                                                        distance_table=cross_distance_table,
        #                                                        k=k_cross,
        #                                                        include_feat_diff=False)
        
        cross_graph = Data(edges_in=in_graph.edge_attr,
                           vert_in_features=in_graph.x,
                           vert_in=in_graph.normed_coords,
                           vert_out_features=out_graph.x,
                           vert_out_gps=out_graph.gps,
                           year=int(out_graph.source.split('_')[-1].split('.')[0]),
                           vert_out=out_graph.normed_coords,
                           edges_out=out_graph.edge_attr,
                           cross_edge_idx=cross_edge_indices,
                           sigmas=out_graph.sigmas,
                           input_affinities=out_graph.input_affinities,
                           filename=out_graph.source,
                           cross_edges=cross_edge_attributes,
                           edge_idx_in=in_graph.edge_index,
                           edge_idx_out=out_graph.edge_index,
                           vert_in_gps=in_graph.gps)
        
        return cross_graph
    
    def cross_graph_sampling(self, n_samples:int=None, perplexity:int=None, k_cross:int=None,
                             N_in:int=None, k_in:int=None, input_sampling_method:str="random",
                             N_out:int=None, k_out:int=None, output_sampling_method:str="regular_subset"):
        graph_list = []
        
        out_graph=self.sample_graph(N=N_out, k=k_out, perplexity=perplexity, sampling_method=output_sampling_method)
        print('output graph done!')
        for s in range(n_samples):
            print(f'{s+1}/{n_samples}')
            in_graph = self.sample_graph(N=N_in, k=k_in, perplexity=perplexity, sampling_method=input_sampling_method)
            cross_graph = self.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=k_cross)
            graph_list.append(cross_graph)
        
        return graph_list

# %%


Generator = AlphaEarth_Graph_Generator(sample_config) 
Generator.load_subregion()
graph1 = Generator.sample_graph(N=1000, k=10, perplexity=20, sampling_method="random")
graph2 = Generator.sample_graph(N=1000, k=10, perplexity=20, sampling_method="regular_subset")

# %%


Generator = AlphaEarth_Graph_Generator(sample_config) 
Generator.load_subregion()
#graph = Generator.sample_graph(N=1000, k=10, perplexity=20, sampling_method="regular_subset")

subgraph_list = Generator.cross_graph_sampling(n_samples=100, perplexity=20, k_cross=10,
                             N_in=1000, k_in=10, input_sampling_method="random",
                             N_out=2000, k_out=10, output_sampling_method="regular_subset")


























































# %%

dataset_config = {
    "data_folder_path": "/Volumes/T7-storage/ERA5/data/",
    "k": 5,
    "k_out": 5,
    "k_cross": 3,
    "perplexity": 15,
    "n_points_input_min": 1000,
    "n_points_input_max": 2000,
    "n_points_lat_out": 100,
    "n_points_lon_out": 100,
    "train_samples_per_year": 100,
    "test_samples_per_year": 100,
    "lat_min": 59.66,
    "lat_max": 59.73,
    "lon_min": 10.79,
    "lon_max": 10.86,
    "train_years": [2018, 2019, 2020],
    "test_years": [2021],
    "cache_region_data": True
    }


sample_config = {
    "data_folder_path": "/Volumes/T7-storage/alphaEarth/data",
    "filename": "sat_emb_2018.tif",
    "bbox": (10.79, 59.66, 10.86, 59.73),
    "include_feat_diff":True,
    "nodeFeat_include_pos":True
}

cross_graph_config = {
    
}

# %% Associated workflow:
# TODO: this should be summarized to an additional funciton in the end
Generator = AlphaEarth_Graph_Generator(sample_config) 
Generator.load_subregion()
#graph = Generator.sample_graph(N=1000, k=10, perplexity=20, sampling_method="regular_subset")

subgraph_list = Generator.cross_graph_sampling(n_samples=10, perplexity=20, k_cross=10,
                             N_in=1000, k_in=10, input_sampling_method="random",
                             N_out=2000, k_out=10, output_sampling_method="regular_subset")







# %% test cross_knn
Generator = AlphaEarth_Graph_Generator(sample_config) 
Generator.load_subregion()
graph_A = Generator.sample_graph(N=10, k=2, perplexity=20, sampling_method="random")
graph_B = Generator.sample_graph(N=10, k=2, perplexity=20, sampling_method="regular_subset")
A=graph_A.normed_coords
B=graph_B.normed_coords


def cross_knn(A, B, k):
    # 1. Build index
    index = faiss.IndexFlatL2(A.shape[1])  # L2 distance
    index.add(A)  # add reference points

    # 2. Search
    D, I = index.search(B, k)

    return D, I

D, I = cross_knn(A = A, B = B, k=2)





#%%


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_knn_edges(A: np.ndarray, B: np.ndarray, I: np.ndarray, D: np.ndarray = None, figsize=(10, 8)):
    """
    Plot two 3D point clouds (A and B) and draw edges from each B point
    to its k nearest neighbors in A using the precomputed KNN indices.

    Args:
        A (np.ndarray): Reference points, shape (NA, 3)
        B (np.ndarray): Query points, shape (NB, 3)
        I (np.ndarray): Indices of k nearest neighbors in A for each B, shape (NB, k)
        D (np.ndarray, optional): Distances to neighbors, shape (NB, k). If provided,
                                  edge color intensity can be scaled by distance.
        figsize (tuple): Matplotlib figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot A
    ax.scatter(A[:,0], A[:,1], A[:,2], color='blue', label='A', alpha=0.6, s=50)
    
    # Plot B
    ax.scatter(B[:,0], B[:,1], B[:,2], color='red', label='B', alpha=0.8, s=50)
    
    # Draw edges from B to neighbors in A
    NB, k = I.shape
    for b_idx in range(NB):
        b_point = B[b_idx]
        for neighbor_rank, a_idx in enumerate(I[b_idx]):
            a_point = A[a_idx]
            xs, ys, zs = zip(b_point, a_point)
            
            if D is not None:
                # Optional: scale alpha by distance
                # shorter edges darker, longer edges lighter
                max_dist = np.max(D)
                alpha = 1.0 - (D[b_idx, neighbor_rank] / max_dist) * 0.8
                alpha = np.clip(alpha, 0.1, 1.0)
            else:
                alpha = 0.5
            
            ax.plot(xs, ys, zs, color='gray', alpha=alpha)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()



# %%

def plot_cross_knn_graph(A: np.ndarray, B: np.ndarray, edges: np.ndarray, title="Cross KNN Graph"):
    """
    Plot two point clouds A and B in 3D and draw edges from B to their neighbors in A.

    Parameters
    ----------
    A : np.ndarray, shape (N_a, 3)
        Reference points.
    B : np.ndarray, shape (N_b, 3)
        Query points.
    edges : np.ndarray, shape (2, M)
        Each column [i, j] means B[i] -> A[j].
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='blue', label='A', s=20, alpha=0.7)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='red', label='B', s=20, alpha=0.7)
    
    # Draw edges
    for b_idx, a_idx in edges.T:  # edges shape = (2, M)
        b_point = B[int(b_idx)]
        a_point = A[int(a_idx)]
        xs, ys, zs = [b_point[0], a_point[0]], [b_point[1], a_point[1]], [b_point[2], a_point[2]]
        ax.plot(xs, ys, zs, c='gray', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
Generator = AlphaEarth_Graph_Generator(sample_config) 
Generator.load_subregion()
subgraph_list = Generator.cross_graph_sampling(n_samples=1, perplexity=20, k_cross=2,
                             N_in=10, k_in=3, input_sampling_method="random",
                             N_out=20, k_out=3, output_sampling_method="regular_subset")

cross_graph = subgraph_list[0]



plot_cross_knn_graph(A=cross_graph.vert_in, B=cross_graph.vert_out, edges=cross_graph.cross_edge_idx)




# %% subgraph workflow
Generator = AlphaEarth_Graph_Generator(sample_config) 
# 1. sample output graph with or without channel interpolation (store it, e.g. skip creating it when alreay exists)
# 2. sample input graph 
# 3. establish cross connections: knn, edges, edge_attr, ...




# %%
class GraphLMDBWriter:
    def __init__(self, db_path, map_size=int(1e12)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)
        
        with self.env.begin(write=True) as txn:
            raw_len = txn.get(b'__len__')
            self.length = int(raw_len.decode()) if raw_len else 0

    def append(self, data: Data):
        with self.env.begin(write=True) as txn:
            txn.put(str(self.length).encode(), pickle.dumps(data))
            self.length += 1
            txn.put(b'__len__', str(self.length).encode())

    def close(self):
        self.env.close()
# %%
def create_dataset(data_folder, dates, n_samples, n_points):
    for BLABLANBLA:
        sam
# %%

