#%%
import sys
import os
from typing import Tuple, Optional, Union
import torch
import rasterio
from rasterio.windows import from_bounds
import xarray as xr
import numpy as np
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


    def generate_normed_sample(self, N: int, method: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate a sample from initially loaded data self.da.
        
        :param N: number of points to sample
        :type N: int
        :param method: sampling method - one of:
            'random': randomly samples N points
            'interpolate': not defined yet
            'regular_subset': selects a subset matching a regular grid structure
        :type method: str
        :returns: (normed_values, gps) - tensor of normalized values and numpy array of GPS coordinates
        :rtype: Tuple[torch.Tensor, np.ndarray]
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



    def cartesian_projection(self, lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> np.ndarray:
        """
        Convert GPS coordinates (lat/lon) to ECEF coordinates (x, y, z).
        Vectorized over all points. CPU-only (pyproj constraint).
        
        :param lat: latitude value(s)
        :param lon: longitude value(s)
        :returns: Cartesian coordinates of shape (N, 3)
        :rtype: np.ndarray
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


    def norm_coords(self, cartesian: np.ndarray) -> np.ndarray:
        """
        Normalize 3-dim cartesian coordinates to range approximately [-1, 1].
        The scale of the 3 dimensions w.r.t each other remains the same.
        This is accomplished by subtracting their individual center values, but dividing by a common denominator.
        
        Requires subregion to be loaded via 'self.load_subregion()'
        
        :param cartesian: Cartesian coordinates of shape (N, 3)
        :type cartesian: np.ndarray
        :returns: Normalized coordinates of shape (N, 3)
        :rtype: np.ndarray
        """
        bound_1 = self.cartesian_projection(np.min(self.grid_latitudes), np.min(self.grid_longitudes))
        bound_2 = self.cartesian_projection(np.min(self.grid_latitudes), np.max(self.grid_longitudes))
        bound_3 = self.cartesian_projection(np.max(self.grid_latitudes), np.min(self.grid_longitudes))
        bound_4 = self.cartesian_projection(np.max(self.grid_latitudes), np.max(self.grid_longitudes))
        
        bounds = np.vstack([bound_1, bound_2, bound_3, bound_4])
        
        x_min, y_min, z_min = np.min(bounds, axis=0)
        x_max, y_max, z_max = np.max(bounds, axis=0)
        
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        
        denominator = np.sqrt(3) * np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
        
        # Create a copy to avoid mutating the input array
        normed_coords = cartesian.copy()
        normed_coords[:, 0] = (cartesian[:, 0] - x_center) / denominator
        normed_coords[:, 1] = (cartesian[:, 1] - y_center) / denominator
        normed_coords[:, 2] = (cartesian[:, 2] - z_center) / denominator
        
        return normed_coords


    def get_edges(self, source: np.ndarray, target: np.ndarray, k: int, mode: str = "self_search") -> Tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
        """
        Search for k nearest neighbors of 'target' in 'source' using FAISS.
        CPU-only (FAISS constraint).
        
        :param source: dataset to find neighbors in, shape (N, D)
        :type source: np.ndarray
        :param target: dataset with points that neighbors are searched for, shape (M, D)
        :type target: np.ndarray
        :param k: number of neighbors
        :type k: int
        :param mode: 'self_search' (excludes self) or 'cross_search'
        :type mode: str
        :returns: (distance_table, neighbor_table, edge_indices, edge_dist)
        :rtype: Tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray]
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
        # more efficient. previously: edge_indices = torch.tensor(np.vstack([source_idx, target_idx]))
        edge_indices = torch.from_numpy(np.vstack([source_idx, target_idx]))
        
        
        edge_dist = distances.reshape(-1, 1)
        
        distance_table, neighbor_table = distances, indices # renaming to match other method
        
        return distance_table, neighbor_table, edge_indices, edge_dist


    def get_edge_attributes(self, source_coords: torch.Tensor, 
                            target_coords: torch.Tensor, 
                            edge_indices: torch.Tensor, 
                            edge_dist: torch.Tensor,
                            distance_table: torch.Tensor,
                            k: int,
                            include_feat_diff: bool = False):
        """
        Compute edge attributes for graph edges. Runs on GPU if inputs are on GPU.
        
        :param source_coords: normed_coords of source graph
        :type source_coords: torch.Tensor
        :param target_coords: normed_coords of target graph
        :type target_coords: torch.Tensor
        :param edge_indices: edge index tensor of shape (2, E)
        :type edge_indices: torch.Tensor
        :param edge_dist: edge distances of shape (E, 1) or (E,)
        :type edge_dist: torch.Tensor
        :param distance_table: distance table of shape (N, k)
        :type distance_table: torch.Tensor
        :param k: number of neighbors
        :type k: int
        :param include_feat_diff: whether to include feature differences (not implemented)
        :type include_feat_diff: bool
        """
        
        # target node positions
        node_pos = target_coords[edge_indices[1, :]]
        
        # regularized distances: divide by max neighbor distance per node
        max_neighbor_dists = distance_table[:, k-1].repeat_interleave(k)
        edge_dist_squeezed = edge_dist.squeeze() if edge_dist.dim() > 1 else edge_dist
        regularized_distances = (edge_dist_squeezed / max_neighbor_dists).unsqueeze(1)
        
        # normed displacement vector
        source_nodes = source_coords[edge_indices[0, :]]
        target_nodes = target_coords[edge_indices[1, :]]
        absolute_displacement = source_nodes - target_nodes
        epsilon = 1e-12
        normed_displacement = absolute_displacement / (torch.norm(absolute_displacement, dim=1, keepdim=True) + epsilon)
        
        # NOTE optionally include date encoding, etc.
        if include_feat_diff:
            print('including feature difference not yet implemented. Continuing without')
        
        # concatenate all edge attributes
        edge_attributes = torch.cat([node_pos, normed_displacement, regularized_distances], dim=1)
        
        return edge_attributes.to(torch.float32)


    def get_node_features(self, normed_coords: torch.Tensor, normed_values: Optional[torch.Tensor] = None, mode: str = "channel_only") -> torch.Tensor:
        """
        Construct node features from coordinates and/or values. GPU-compatible.
        
        :param normed_coords: Normalized coordinates tensor of shape (N, 3)
        :type normed_coords: torch.Tensor
        :param normed_values: Normalized channel values tensor of shape (N, F)
        :type normed_values: torch.Tensor, optional
        :param mode: 'channel_only', 'position_only', or 'channel_and_position'
        :type mode: str
        :returns: Node features tensor
        :rtype: torch.Tensor
        """
        # Ensure normed_coords is a tensor on the correct device
        if not isinstance(normed_coords, torch.Tensor):
            normed_coords = torch.as_tensor(normed_coords, dtype=torch.float32)
        
        if mode == "channel_only":
            node_features = normed_values
        elif mode == "position_only":
            node_features = normed_coords
        elif mode == "channel_and_position":
            node_features = torch.cat([normed_values, normed_coords], dim=1)
        else:
            raise ValueError(f'Invalid node feature mode: {mode}')
            
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


    def old_get_sigmas(self, distance_table:np.ndarray=None, perplexity:int=None):
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


    def get_sigmas(self, distance_table: np.ndarray, perplexity: float):
        target_entropy = np.log(perplexity)
        sigmas = []

        for dists in distance_table:
            dists = np.array(dists, dtype=np.float64)

            # Use mean neighbor distance as scale
            mean_dist = np.mean(dists)
            sigma_min = 1e-3 * mean_dist  # small fraction
            sigma_max = 10 * mean_dist    # large fraction
            sigma = mean_dist             # initial guess

            for _ in range(50):
                # Gaussian kernel
                p = np.exp(-dists**2 / (2 * sigma**2))
                p /= np.sum(p)
                entropy = -np.sum(p * np.log(p + 1e-10))

                if abs(entropy - target_entropy) < 1e-5:
                    break

                # binary search
                if entropy < target_entropy:
                    # entropy too low -> sigma too small -> increase sigma
                    sigma_min = sigma
                    sigma = (sigma + sigma_max) / 2
                else:
                    # entropy too high -> sigma too large -> decrease sigma
                    sigma_max = sigma
                    sigma = (sigma + sigma_min) / 2

            sigmas.append(sigma)
        
        # TODO: FIX this:
        ones = np.ones_like(sigmas)

        return np.array(ones)




    def get_affinities(self, normed_values: torch.Tensor, edge_indices: torch.Tensor, sigmas: torch.Tensor):
        """
        Compute local affinity matrix using per-point sigma values. Runs on GPU if inputs are on GPU.

        Args:
            normed_values (Tensor): [N, D] input features
            edge_indices (LongTensor): [2, E] edge list, edges[0,:] = source, edges[1,:] = target
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
        
        return p  # input affinities


    def sample_graph(self, N: int = None, k: int = None, perplexity: int = None, sampling_method: str = "random", gen_feature_mode: str = "channel_only"):
        # ---- CPU operations (data loading, pyproj, FAISS) ----
        normed_values, gps = self.generate_normed_sample(N=N, method=sampling_method)  # sample subset 
        cartesian = self.cartesian_projection(lat=gps[:, 0], lon=gps[:, 1])  # pyproj: CPU only
        normed_coords_np = self.norm_coords(cartesian=cartesian)  # norm coordinates based on subregion
        distance_table_np, neighbor_table, edge_indices, edge_dist = self.get_edges(source=normed_coords_np, target=normed_coords_np, k=k, mode="self_search")  # FAISS: CPU only
        
        # get_sigmas requires numpy distance_table (iterative binary search, not GPU-friendly)
        sigmas_np = self.get_sigmas(distance_table=distance_table_np, perplexity=perplexity)
        
        # ---- GPU boundary (ONE transfer) ----
        normed_values = normed_values.to(device)
        normed_coords = torch.as_tensor(normed_coords_np, dtype=torch.float32, device=device)
        edge_indices = edge_indices.to(device)
        edge_dist = torch.as_tensor(edge_dist, dtype=torch.float32, device=device)
        distance_table = torch.as_tensor(distance_table_np, dtype=torch.float32, device=device)
        sigmas = torch.as_tensor(sigmas_np, dtype=torch.float32, device=device)
        
        # ---- GPU operations ----
        edge_attributes = self.get_edge_attributes(
            source_coords=normed_coords, 
            target_coords=normed_coords, 
            edge_indices=edge_indices, 
            edge_dist=edge_dist,
            distance_table=distance_table,
            k=k,
            include_feat_diff=False
        )
        node_features = self.get_node_features(normed_coords=normed_coords, normed_values=normed_values, mode=gen_feature_mode)
        input_affinities = self.get_affinities(normed_values=normed_values, edge_indices=edge_indices, sigmas=sigmas)
        
        sample = Data(
            x=node_features, 
            edge_index=edge_indices, 
            edge_attr=edge_attributes, 
            edge_dist=edge_dist,
            input_affinities=input_affinities, 
            sigmas=sigmas, 
            gps=gps, 
            normed_coords=normed_coords, 
            normed_values=normed_values, 
            year=self.year
        )
        return sample



    def connect_graphs(self, in_graph: Data, out_graph: Data, k_cross: int):
        """
        This method creates a cross graph structure connecting input and output graphs.
        
        The input/output graphs come from sample_graph() with tensors on GPU.
        FAISS requires CPU numpy arrays, so we extract them, run FAISS, then convert back to GPU.
        
        returns Data object with cross-graph connections
        """
        # ---- Extract CPU arrays for FAISS (CPU-only constraint) ----
        in_coords_np = in_graph.normed_coords.cpu().numpy() if isinstance(in_graph.normed_coords, torch.Tensor) else in_graph.normed_coords
        out_coords_np = out_graph.normed_coords.cpu().numpy() if isinstance(out_graph.normed_coords, torch.Tensor) else out_graph.normed_coords
        
        # FAISS nearest neighbor search (CPU)
        cross_distance_table_np, cross_neighbor_table, cross_edge_indices, cross_edge_dist_np = self.get_edges(
            source=in_coords_np, 
            target=out_coords_np, 
            k=k_cross, 
            mode="cross_search"
        )
        
        # ---- Convert to GPU tensors ----
        cross_edge_indices = cross_edge_indices.to(device)
        cross_edge_dist = torch.as_tensor(cross_edge_dist_np, dtype=torch.float32, device=device)
        cross_distance_table = torch.as_tensor(cross_distance_table_np, dtype=torch.float32, device=device)
        
        # Ensure graph coords are on GPU for get_edge_attributes
        in_normed_coords = in_graph.normed_coords if in_graph.normed_coords.device == device else in_graph.normed_coords.to(device)
        out_normed_coords = out_graph.normed_coords if out_graph.normed_coords.device == device else out_graph.normed_coords.to(device)
        
        # ---- GPU operations ----
        cross_edge_attributes = self.get_edge_attributes(
            source_coords=in_normed_coords, 
            target_coords=out_normed_coords, 
            edge_indices=cross_edge_indices, 
            edge_dist=cross_edge_dist,
            distance_table=cross_distance_table,
            k=k_cross,
            include_feat_diff=False
        )
        
        cross_graph = Data(
            vert_in_features=in_graph.x,                 # node features of input graph (= node position and/or channel values)
            edges_in=in_graph.edge_attr,                 # edge attributes of input graph
            edge_idx_in=in_graph.edge_index,             # edge indices for input graph
            vert_in=in_graph.normed_coords,              # node positions of input graph
            vert_in_gps=in_graph.gps,                    # input graphs gps coordinates
            vert_in_values=in_graph.normed_values,       # pure channel values of input graph 
            edge_dist_in=in_graph.edge_dist,             # node distance per edge for input graph
            edges_out=out_graph.edge_attr,               # edge attributes of output graph
            edge_idx_out=out_graph.edge_index,           # edge indices for output graph
            vert_out=out_graph.normed_coords,            # node positions of output graph
            vert_out_gps=out_graph.gps,                  # output graphs gps coordinates
            vert_out_values=out_graph.normed_values,     # pure channel values of output graph (GROUND TRUTH)
            edge_dist_out=out_graph.edge_dist,           # node distance per edge for output graph
            cross_edges=cross_edge_attributes,           # edge attributes for cross connections
            cross_edge_idx=cross_edge_indices,           # edge indices for cross connections
            cross_edge_dist=cross_edge_dist,             # node distance per edge for cross connections
            sigmas=out_graph.sigmas,                     # sigma values per point for output nodes
            input_affinities=out_graph.input_affinities, # Ground truth data --> into loss function
            year=out_graph.year                          # year of origin data
        )
        
        return cross_graph
    
    
    def construct_output_node_features(self, cross_graph: Data, interpolate: bool = True, gen_feature_mode: str = "channel_only") -> torch.Tensor:
        """
        Construct node features for the output graph. GPU-compatible.
        
        :param cross_graph: Cross graph data object
        :type cross_graph: Data
        :param interpolate: If true, interpolates channel values for target graph based on neighboring values from source graph
        :type interpolate: bool
        :param gen_feature_mode: 'channel_only', 'position_only', or 'channel_and_position'
        :type gen_feature_mode: str
        :returns: Constructed node features tensor
        :rtype: torch.Tensor
        """
        # Data is already on GPU from connect_graphs, just ensure tensor type
        normed_values_source = cross_graph.vert_in_values
        normed_coords_source = cross_graph.vert_in if isinstance(cross_graph.vert_in, torch.Tensor) else torch.as_tensor(cross_graph.vert_in, device=device)
        normed_coords_target = cross_graph.vert_out if isinstance(cross_graph.vert_out, torch.Tensor) else torch.as_tensor(cross_graph.vert_out, device=device)
        cross_edge_idx = cross_graph.cross_edge_idx
        
        if interpolate and (gen_feature_mode != "position_only"):
            interpolated_values = self.inverse_distance_interpolation(
                normed_values_source=normed_values_source,
                normed_coords_source=normed_coords_source,
                normed_coords_target=normed_coords_target,
                cross_edge_idx=cross_edge_idx,
                eps=1e-12
            )
            constructed_features = self.get_node_features(
                normed_coords=normed_coords_target, 
                normed_values=interpolated_values, 
                mode=gen_feature_mode
            )
        else:
            constructed_features = self.get_node_features(
                normed_coords=normed_coords_target, 
                mode=gen_feature_mode
            )
        
        return constructed_features.to(torch.float32)
    
    
    def cross_graph_sampling(self, n_samples:int=None, perplexity:int=None, k_cross:int=None, interpolate_graph_features:bool=True, gen_feature_mode:str="channel_only",
                             N_in:int=None, k_in:int=None, input_sampling_method:str="random",
                             N_out:int=None, k_out:int=None, output_sampling_method:str="regular_subset"):
        graph_list = []
        
        out_graph=self.sample_graph(N=N_out, k=k_out, perplexity=perplexity, sampling_method=output_sampling_method)
        print('output graph done!')
        for s in range(n_samples):
            print(f'{s+1}/{n_samples}')
            in_graph = self.sample_graph(N=N_in, k=k_in, perplexity=perplexity, sampling_method=input_sampling_method)
            cross_graph = self.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=k_cross)
            cross_graph["vert_out_features_constructed"] = self.construct_output_node_features(cross_graph=cross_graph, interpolate=interpolate_graph_features, gen_feature_mode=gen_feature_mode)
            graph_list.append(cross_graph)
        return graph_list


# %%
