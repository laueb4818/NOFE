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


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'using Device: {device}')

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
        print(f'\nloading data from {self.file_path} ...')
        
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



    def get_edges(self, source:np.ndarray=None, target:np.ndarray=None, k:int=None, perplexity:int=0, mode:str="self_search"):
        """
        Searches for k nearest neighbors of 'target' in 'source'.
        
        :param source: dataset to find neighbors in (coordinates)
        :type source: np.ndarray
        :param target: dataset with points that neighbors are searched for (coordinates)
        :type target: np.ndarray
        :param k: number of neighbors
        :type k: int
        :param cross_search: determines operations
        :type cross_search: bool
        """
        
        if mode=="self_search":
            NN = max(k+1, perplexity+1)
        else:
            NN = max(k, perplexity+1)


        
        N, D = source.shape
        index = faiss.IndexFlatL2(D)  # L2 distance index (Euclidean)
        index.add(source)
        distance_tab, indices_tab = index.search(target, NN) 


        # get long table of edge indices and edge distances
        if mode == "cross_search":
            dist_tab_k, idx_tab_k = distance_tab[:, :k], indices_tab[:, :k]       
        elif mode == "self_search": 
            dist_tab_k, idx_tab_k = distance_tab[:, 1:k+1], indices_tab[:, 1:k+1]       
        else:
            print("no valid mode was provided!")
        
        source_idx = idx_tab_k.reshape(-1)
        target_idx = np.repeat(np.arange(target.shape[0]),k)
        edge_indices = torch.from_numpy(np.vstack([source_idx, target_idx]))
        
        edge_dist = dist_tab_k.reshape(-1, 1)
       
        # get neighborhood index table for sigma calulation
        idx_tab_p = indices_tab[:, 1:perplexity+1]


        return edge_indices, edge_dist, idx_tab_p
    
    
    
    def feature_distances(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distances for neighboring node pairs.

        Args:
        indices: LongTensor of shape (N, p) specifying neighbor indices for each node.
        values: FloatTensor of shape (N, f) with coordinates for each node.

        Returns:
        distances: FloatTensor of shape (N, p) with Euclidean distances to neighbors.
        """
        # Gather neighbor coordinates
        # indices: (N, p), values: (N, f) -> neighbor_values: (N, p, f)
        neighbor_values = values[indices] # advanced indexing in PyTorch

        # Expand original values to (N, 1, f) to broadcast
        node_values = values.unsqueeze(1) # (N, 1, f)

        # Compute Euclidean distance along last axis (f)
        distances = torch.norm(node_values - neighbor_values, dim=-1) # (N, p)

        return distances


    def optimize_sigma(self, distances: torch.Tensor, target_perplexity: float,
        tol=1e-5, max_iter=50) -> torch.Tensor:
        """
        Optimize sigma for each point to match target perplexity.

        Args:
        distances: FloatTensor of shape (N, p) with squared distances to neighbors.
        target_perplexity: desired perplexity value (scalar).
        tol: tolerance for perplexity matching.
        max_iter: max binary search iterations.

        Returns:
        sigmas: FloatTensor of shape (N,) with optimized sigma values.
        """
        N, p = distances.shape
        # Convert distances to float32 for stability
        distances = distances.float()

        # Compute log2(target_perplexity)
        log_perplexity = torch.log2(torch.tensor(target_perplexity, device=distances.device))

        # Initialize sigma bounds
        sigma_min = torch.zeros(N, device=distances.device)
        sigma_max = torch.full((N,), 1e5, device=distances.device)
        sigma = torch.full((N,), 1.0, device=distances.device)

        for _ in range(max_iter):
            # Compute conditional probabilities
            # p_ij ~ exp(-d_ij^2 / 2σ^2)
            # sigma: (N,) -> (N,1) for broadcasting
            p_ij = torch.exp(-distances / (2 * sigma.unsqueeze(1)**2))
            # Normalize
            p_ij_sum = p_ij.sum(dim=1, keepdim=True)
            p_ij = p_ij / p_ij_sum

            # Compute entropy H(P_i)
            H = -(p_ij * torch.log2(p_ij + 1e-12)).sum(dim=1) # shape (N,)

            # Binary search update
            mask = H > log_perplexity # need larger sigma
            sigma_min = torch.where(mask, sigma, sigma_min)
            sigma_max = torch.where(mask, sigma_max, sigma)
            # Avoid sigma_max == 1e5 for the first iteration
            sigma = (sigma_min + sigma_max) / 2

            # Early stopping if all entropies within tolerance
            if torch.all(torch.abs(H - log_perplexity) < tol):
                break

        return sigma

    def get_edge_attributes(
        self,
        source_coords: torch.Tensor,    # (N, f)
        target_coords: torch.Tensor,    # (N, f)
        edge_indices: torch.Tensor,     # (2, E), dtype=torch.long
        edge_dist: torch.Tensor,        # (E, 1)
        k: int,
        include_feat_diff: bool = False
    ):
        """
        Returns:
            edge_attributes: (E, 3f + 1)
        """

        # Node positions (target)
        node_pos = target_coords[edge_indices[1]]  # (E, f)

        # ---- Regularized distances ----
        # max dist per source node (repeat k times)
        # edge_indices[0] tells us which node each edge belongs to
        max_dist_per_node = torch.zeros(
            source_coords.shape[0], device=edge_dist.device
        )

        max_dist_per_node.scatter_reduce_(
            0, edge_indices[0], edge_dist.squeeze(), reduce="amax"
        )

        regularized_distances = (
            edge_dist.squeeze() /
            max_dist_per_node[edge_indices[0]]
        ).unsqueeze(1)

        # ---- Displacement vectors ----
        source_nodes = source_coords[edge_indices[0]]
        target_nodes = target_coords[edge_indices[1]]
        absolute_displacement = source_nodes - target_nodes

        norm = torch.norm(absolute_displacement, dim=1, keepdim=True)
        normed_displacement = absolute_displacement / (norm + 1e-12)

        if include_feat_diff:
            print("including feature difference not yet implemented. Continuing without")

        # ---- Stack features ----
        edge_attributes = torch.cat(
            [node_pos, normed_displacement, regularized_distances],
            dim=1
        )

        return edge_attributes


    def get_node_features(self, normed_coords:np.ndarray=None, normed_values:torch.Tensor=None, mode:str="channel_only"):
        if type(normed_coords) != torch.Tensor:
            normed_coords = torch.as_tensor(normed_coords, dtype=torch.float32)
        
        if mode=="channel_only":
            node_features = normed_values
        elif mode=="position_only":
            node_features=normed_coords
        elif mode=="channel_and_position":
            node_features = torch.cat([normed_values, normed_coords], dim=1)
        else:
            print('invalid node feature mode')
            
        return node_features


    def inverse_distance_interpolation(self,
        normed_values_source: torch.Tensor,   # (N, F)
        normed_coords_source: torch.Tensor,   # (N, D)
        normed_coords_target: torch.Tensor,   # (M, D)
        cross_edge_idx: torch.Tensor,         # (2, E)
        eps: float = 1e-12
    ):
        """
        Interpolate source features to target points using inverse distance weighting.
        
        Returns:
            v_target: (M, F) interpolated features at target points
        """
        # 1. Extract source and target indices
        source_idx = cross_edge_idx[0]
        target_idx = cross_edge_idx[1]
        
        # 2. Compute distances for each edge
        diff = normed_coords_target[target_idx] - normed_coords_source[source_idx]  # (E, D)
        dist = torch.norm(diff, dim=1) + eps  # avoid division by zero, shape (E,)
        
        # 3. Compute weights
        weights = 1.0 / dist  # (E,)
        
        # 4. Weighted sum of source features
        weighted_values = normed_values_source[source_idx] * weights[:, None]  # (E, F)
        v_target = scatter(weighted_values, target_idx, dim=0, reduce='sum')  # sum over neighbors
        
        # 5. Normalize by sum of weights
        sum_weights = scatter(weights, target_idx, dim=0, reduce='sum')  # (M,)
        v_target /= sum_weights[:, None]
        
        return v_target


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
        p = p.to(torch.float32)
        
        return p #input affinities


    def sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random", gen_feature_mode:str="channel_only"):
        normed_values, gps = self.generate_normed_sample(N=N, method=sampling_method) # sample subset 
        cartesian = self.cartesian_projection(lat=gps[:, 0], lon=gps[:, 1]) # get 3 dim. cartesian coordinates for sample locations
        normed_coords = self.norm_coords(cartesian=cartesian) # norm coordinates based on subregion
        edge_indices, edge_dist, idx_tab_p = self.get_edges(source=normed_coords, target=normed_coords, k=k, perplexity=perplexity, mode="self_search")

        # --- GPU transition ---
        edge_indices = torch.as_tensor(edge_indices, device=device, dtype=torch.long) 
        edge_dist = torch.as_tensor(edge_dist, device=device, dtype=torch.float32)
        idx_tab_p = torch.as_tensor(idx_tab_p, device=device, dtype=torch.long)
        normed_values = torch.as_tensor(normed_values, device=device, dtype=torch.float32)
        normed_coords = torch.as_tensor(normed_coords, device=device, dtype=torch.float32)

        edge_attributes = self.get_edge_attributes(source_coords=normed_coords, 
                            target_coords=normed_coords, 
                            edge_indices=edge_indices, 
                            edge_dist=edge_dist,
                            k=k,
                            include_feat_diff=False)
        node_features = self.get_node_features(normed_coords=normed_coords, normed_values=normed_values, mode=gen_feature_mode)
        if perplexity is not None and perplexity != 0:
            feature_distances = self.feature_distances(indices=idx_tab_p, values=normed_values)
            sigmas = self.optimize_sigma(distances=feature_distances, target_perplexity=perplexity)
            input_affinities = self.get_affinities(normed_values=normed_values, edge_indices=edge_indices, sigmas=sigmas)
        else:
            input_affinities, sigmas = None, None
        return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, edge_dist=edge_dist ,input_affinities=input_affinities, sigmas=sigmas, gps=gps, normed_coords=normed_coords, normed_values=normed_values, year=self.year)



    def connect_graphs(self, in_graph:Data=None, out_graph:Data=None, k_cross:int=None):
        """
        This method creates a cross graph structure.
        the output is something like
        out_graph = sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random")
        
        returns list of Data objects
        !!! Careful when writing data as it is in form of a list !!!
        """
        cross_edge_indices, cross_edge_dist, _  = self.get_edges(source=in_graph.normed_coords.cpu().numpy(), 
                                                                                 target=out_graph.normed_coords.cpu().numpy(), 
                                                                                 k=k_cross, mode="cross_search", perplexity=0)
        cross_edge_indices = torch.as_tensor(cross_edge_indices, device=device, dtype=torch.long)
        cross_edge_dist = torch.as_tensor(cross_edge_dist, device=device, dtype=torch.float32)
        
        cross_edge_attributes = self.get_edge_attributes(source_coords=in_graph.normed_coords, 
                            target_coords=out_graph.normed_coords, 
                            edge_indices=cross_edge_indices, 
                            edge_dist=cross_edge_dist,
                            k=k_cross,
                            include_feat_diff=False,
                            )
        
        cross_graph = Data(vert_in_features=in_graph.x,             # node features of input graph (= node position and/or channel values)
                           edges_in=in_graph.edge_attr,             # edge attributes of input graph
                           edge_idx_in=in_graph.edge_index,         # edge indices for input graph
                           vert_in=in_graph.normed_coords,          # node positions of input graph
                           vert_in_gps=in_graph.gps,                # input graphs gps coordinates
                           vert_in_values=in_graph.normed_values,   # pure channel values of input graph 
                           edge_dist_in=in_graph.edge_dist,         # node distance per edge for input graph
                           #vert_out_features_truth=out_graph.x,    # ground truth for corresponding input node feature BUT not used because for loss, we only want channel values
                           edges_out=out_graph.edge_attr,           # edge attributes of output graph
                           edge_idx_out=out_graph.edge_index,       # edge indices for output graph
                           vert_out=out_graph.normed_coords,        # node positions of output graph
                           vert_out_gps=out_graph.gps,              # output graphs gps coordinates
                           vert_out_values=out_graph.normed_values, # pure channel values of output graph (THIS IS USED FOR THE GROUND TRUTH)
                           edge_dist_out=out_graph.edge_dist,       # node distance per edge for output graph
                           cross_edges=cross_edge_attributes,       # edge attributes for cross connections between input and output
                           cross_edge_idx=cross_edge_indices,       # edge indices for cross connections between input and output
                           cross_edge_dist=cross_edge_dist,         # node distance per edge for cross connections
                           sigmas=out_graph.sigmas,                 # sigma values per point for output nodes
                           input_affinities=out_graph.input_affinities, # Ground truth data --> into loss function
                           year=out_graph.year)                     # year of origin data
        
        return cross_graph
    
    
    def construct_output_node_features(self, cross_graph:Data, interpolate:bool=True, gen_feature_mode:str="channel_only"):
        """
        construct node features for the output graph
        
        interpolate: if true: interpolates channel values for target graph based on neighboring values from source graph
        gen_feature_mode: determines whether features contain channel values, position or both
        """
        normed_values_source=cross_graph.vert_in_values
        # more efficient. previously: normed_coords_source=torch.tensor(cross_graph.vert_in)
        normed_coords_source=torch.as_tensor(cross_graph.vert_in)
        # more efficient. previously: normed_coords_target=torch.tensor(cross_graph.vert_out)
        normed_coords_target=torch.as_tensor(cross_graph.vert_out)
        cross_edge_idx=cross_graph.cross_edge_idx
        
        if interpolate and (gen_feature_mode != "position_only"):
            interpolated_values = self.inverse_distance_interpolation(
                normed_values_source=normed_values_source,   # (N, F)
                normed_coords_source=normed_coords_source,   # (N, D)
                normed_coords_target=normed_coords_target,   # (M, D)
                cross_edge_idx=cross_edge_idx,         # (2, E)
                eps = 1e-12)
            constructed_features=self.get_node_features(normed_coords=normed_coords_target, normed_values=interpolated_values, mode=gen_feature_mode)
            
        else:
            constructed_features=self.get_node_features(normed_coords=normed_coords_target, mode=gen_feature_mode)
        
        return constructed_features.to(torch.float32)
    
    
    def cross_graph_sampling(self, n_samples:int=None, perplexity:int=None, k_cross:int=None, interpolate_graph_features:bool=True, gen_feature_mode:str="channel_only",
                             N_in:int=None, k_in:int=None, input_sampling_method:str="random",
                             N_out:int=None, k_out:int=None, output_sampling_method:str="regular_subset"):
        graph_list = []
        
        out_graph=self.sample_graph(N=N_out, k=k_out, perplexity=perplexity, sampling_method=output_sampling_method)
        print('output graph done!')
        for s in range(n_samples):
            print(f'{s+1}/{n_samples}')
            in_graph = self.sample_graph(N=N_in, k=k_in, perplexity=0, sampling_method=input_sampling_method)
            cross_graph = self.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=k_cross)
            cross_graph["vert_out_features_constructed"] = self.construct_output_node_features(cross_graph=cross_graph, interpolate=interpolate_graph_features, gen_feature_mode=gen_feature_mode)
            graph_list.append(cross_graph)
        return graph_list


# %%
