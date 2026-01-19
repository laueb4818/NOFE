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


# %% define graph_generator
class AlphaEarth_Graph_Generator:
    """
    This class will load data from alphaEarth data and is equipped with all functionalities to generate graph samples
    """
    def __init__(self, file_path:str=None):
        self.file_path=file_path
        filename = file_path.split('/')[-1]       # 'sat_emb_2018.tif'
        self.year = int(filename.split('_')[-1].split('.')[0])
        
        
        #self.config = sample_config.copy()
        # Dynamically create attributes for easy access
        #for key, value in sample_config.items():
        #    setattr(self, key, value)

    def load_subregion(self, bbox):
        """
        Load a bounding box from a large GeoTIFF and return as xarray.DataArray.
        
        bbox = (min_lon, min_lat, max_lon, max_lat)
        self.da will have structure: (channels, latitude, longitude)
        """
        
        with rasterio.open(self.file_path) as src:
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
        mean = arr.mean(axis=(1,2), keepdims=True)
        std = arr.std(axis=(1,2), keepdims=True)
        self.da = (arr - mean) / std
        self.grid_latitudes = np.linspace(bbox[1], bbox[3], self.da.shape[1])
        self.grid_longitudes = np.linspace(bbox[0], bbox[2], self.da.shape[2])
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


    def sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random", nodeFeat_include_pos:bool=False):
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
        node_features = self.get_node_features(normed_coords=normed_coords, normed_values=normed_values, include_pos=nodeFeat_include_pos)
        sigmas = self.get_sigmas(distance_table=distance_table, perplexity=perplexity)
        input_affinities = self.get_affinities(normed_values=normed_values, edge_indices=edge_indices, sigmas=sigmas)
        return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, input_affinities=input_affinities, sigmas=sigmas, gps=gps, normed_coords=normed_coords, year=self.year)



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
        print(in_graph.normed_coords.shape)
        cross_graph = Data(edges_in=in_graph.edge_attr,
                           vert_in_features=in_graph.x,
                           vert_in=in_graph.normed_coords,
                           vert_out_features=out_graph.x,
                           vert_out_gps=out_graph.gps,
                           year=out_graph.year,
                           vert_out=out_graph.normed_coords,
                           edges_out=out_graph.edge_attr,
                           cross_edge_idx=cross_edge_indices,
                           sigmas=out_graph.sigmas,
                           input_affinities=out_graph.input_affinities,
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
            print(cross_graph.vert_in.shape)
            graph_list.append(cross_graph)
        
        return graph_list

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

#%%

Generator = AlphaEarth_Graph_Generator("/Volumes/T7-storage/alphaEarth/data/sat_emb_2018.tif") 
Generator.load_subregion(bbox=(10.79, 59.66, 10.86, 59.73))
graph1 = Generator.sample_graph(N=1000, k=10, perplexity=20, sampling_method="random")
graph2 = Generator.sample_graph(N=1000, k=10, perplexity=20, sampling_method="regular_subset")

# %%


Generator = AlphaEarth_Graph_Generator("/Volumes/T7-storage/alphaEarth/data/sat_emb_2018.tif") 
Generator.load_subregion(bbox=(10.79, 59.66, 10.86, 59.73))
#graph = Generator.sample_graph(N=1000, k=10, perplexity=20, sampling_method="regular_subset")

subgraph_list = Generator.cross_graph_sampling(n_samples=10, perplexity=20, k_cross=10,
                             N_in=1000, k_in=10, input_sampling_method="random",
                             N_out=2000, k_out=10, output_sampling_method="regular_subset")

#%%




dataset_config_simple = {
    "data_folder_path": "/Volumes/T7-storage/alphaEarth/data/",
    "dataset_save_path": "../datasets/",
    "bbox":(10.79, 59.66, 10.86, 59.73),
    "N": 1000,
    "k": 10,
    "perplexity": 10,
    "sampling_method": "random",
    "train_years": [2018, 2019, 2020],
    "test_years": [2021],
    "train_samples_per_year": 100,
    "test_samples_per_year": 100,
}

dataset_config_subgraphs = {
    
}

def main(config_path):
    """
    Expect 'config_path' as '[config_folder]/[config_id]', e.g. dataset_configs_subgraph/basic
    :param config: Description
    """
    
    # Open and load the JSON
    with open(config_path+".json", "r") as f:
        config = json.load(f)
    config_id = config_path.split('/')[1]
    
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_train.lmdb'):
        print('dataset already exists -- STOP')
        return()
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_test.lmdb'):
        print('dataset already exists -- STOP')
        return()
    
    N=config['N']
    k=config['k']
    perplexity=config['perplexity']
    sampling_method=config['sampling_method']
    
    # Train Set
    print('Generate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    
    for year in config["train_years"]:
        file_path = os.path.join(config["data_folder_path"], "sat_emb_"+str(year)+".tif")
        Generator = AlphaEarth_Graph_Generator(file_path) 
        Generator.load_subregion(bbox=config["bbox"])
        for n in range(config["train_samples_per_year"]):
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method)
            writer.append(graph)
            
            
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for year in config["test_years"]:
        file_path = os.path.join(config["data_folder_path"], "sat_emb_"+str(year)+".tif")
        Generator = AlphaEarth_Graph_Generator(file_path) 
        Generator.load_subregion(bbox=config["bbox"])
        for n in range(config["test_samples_per_year"]):
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method)
            writer.append(graph)




""" Sanity check for edges
# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_3d_points_with_edges(A, B, edges):
    """
    Plots 3D points from sets A and B, and draws edges between them.
    
    Parameters:
        A (np.ndarray): (N,3) array of points
        B (np.ndarray): (M,3) array of points
        edges (torch.Tensor): (2,E) tensor where edges[0,:] are indices in A and edges[1,:] in B
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(A[:,0], A[:,1], A[:,2], c='blue', label='A', s=50)
    ax.scatter(B[:,0], B[:,1], B[:,2], c='red', label='B', s=50)
    
    # Plot edges
    for i in range(edges.shape[1]):
        a_idx = edges[0, i].item()
        b_idx = edges[1, i].item()
        x = [A[a_idx,0], B[b_idx,0]]
        y = [A[a_idx,1], B[b_idx,1]]
        z = [A[a_idx,2], B[b_idx,2]]
        ax.plot(x, y, z, c='gray', alpha=0.7)
    
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# %%
Generator = AlphaEarth_Graph_Generator("/Volumes/T7-storage/alphaEarth/data/sat_emb_2018.tif") 
Generator.load_subregion(bbox=(10.79, 59.66, 10.86, 59.73))

#%%
subgraph_list = Generator.cross_graph_sampling(n_samples=1, perplexity=20, k_cross=2,
                             N_in=20, k_in=2, input_sampling_method="random",
                             N_out=20, k_out=2, output_sampling_method="regular_subset")
cross_graph=subgraph_list[0]

In=cross_graph.vert_in
Out=cross_graph.vert_out
edges=cross_graph.cross_edge_idx

plot_3d_points_with_edges(In, Out, edges)

"""