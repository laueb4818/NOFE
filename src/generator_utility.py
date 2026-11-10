#%%
import os
import torch
from rasterio.windows import from_bounds
import numpy as np
from pyproj import Transformer
import faiss
from torch_geometric.data import Data
from torch_scatter import scatter


# %% 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


import time
import functools
from collections import defaultdict

from utilities import get_output_affinities

def clear_memory_cache():
    """Clear GPU memory cache. Works with CUDA, MPS, and CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

# shared registry
_function_timings = defaultdict(lambda: {"total": 0.0, "calls": 0})


def track_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        stats = _function_timings[func.__name__]
        stats["total"] += elapsed
        stats["calls"] += 1

        return result

    return wrapper

# %% define graph_generator
class Graph_Generator:
    """
    This class is a parent class for the actual generator calsses.
    Actual classes will need to extend this e.g. by loading
    """
    def __init__(self, datafolder:str):
        self.directory= os.path.realpath(datafolder)
        self.eps = 1e-8
    
    @track_time
    def load_data(self):
        # should be implemented by subclass
        pass
    
    @track_time
    def generate_normed_sample(self, N:int, method:str):
        # should be implemented by subclass
        pass
        
        #self.config = sample_config.copy()
        # Dynamically create attributes for easy access
        #for key, value in sample_config.items():
        #    setattr(self, key, value)


    @track_time
    def get_edges(self, source:np.ndarray=None, target:np.ndarray=None, k:int=None, perplexity:int=None, mode:str="self_search"):
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

        if perplexity is None:
            perplexity = 0
        
        # kNN is selected purely by k; perplexity is retained for config compatibility and
        # only matters if the Gaussian-affinity path below is re-enabled.
        #N_per = 3*perplexity
        N_per = k

        if mode=="self_search":
            NN = max(k+1, N_per+1)
        else:
            NN = max(k, N_per+1)


        
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
        idx_tab_p = indices_tab[:, 1:N_per+1]


        return edge_indices, edge_dist, idx_tab_p


    @track_time
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



    @track_time
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
            (max_dist_per_node[edge_indices[0]] + self.eps)
        ).unsqueeze(1)

        # ---- Displacement vectors ----
        source_nodes = source_coords[edge_indices[0]]
        target_nodes = target_coords[edge_indices[1]]
        absolute_displacement = source_nodes - target_nodes

        norm = torch.norm(absolute_displacement, dim=1, keepdim=True)
        normed_displacement = absolute_displacement / (norm + self.eps)

        if include_feat_diff:
            print("including feature difference not yet implemented. Continuing without")

        # ---- Stack features ----
        edge_attributes = torch.cat(
            [node_pos, normed_displacement, regularized_distances],
            dim=1
        )

        return edge_attributes


    @track_time
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


    @track_time
    def inverse_distance_interpolation(self,
        normed_values_source: torch.Tensor,   # (N, F)
        normed_coords_source: torch.Tensor,   # (N, D)
        normed_coords_target: torch.Tensor,   # (M, D)
        cross_edge_index: torch.Tensor,         # (2, E)
    ):
        """
        Interpolate source features to target points using inverse distance weighting.
        
        Returns:
            v_target: (M, F) interpolated features at target points
        """
        # 1. Extract source and target indices
        source_idx = cross_edge_index[0]
        target_idx = cross_edge_index[1]
        
        # 2. Compute distances for each edge
        diff = normed_coords_target[target_idx] - normed_coords_source[source_idx]  # (E, D)
        dist = torch.norm(diff, dim=1) + self.eps  # avoid division by zero, shape (E,)
        
        # 3. Compute weights
        weights = 1.0 / dist  # (E,)
        
        # 4. Weighted sum of source features
        weighted_values = normed_values_source[source_idx] * weights[:, None]  # (E, F)
        v_target = scatter(weighted_values, target_idx, dim=0, reduce='sum')  # sum over neighbors
        
        # 5. Normalize by sum of weights
        sum_weights = scatter(weights, target_idx, dim=0, reduce='sum')  # (M,)
        v_target /= sum_weights[:, None]
        
        return v_target

    @track_time
    def sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random", gen_feature_mode:str="channel_only"):
        normed_values, normed_coords, meta = self.generate_normed_sample(N=N, method=sampling_method)

        edge_indices, edge_dist, idx_tab_p = self.get_edges(source=normed_coords, target=normed_coords, k=k, perplexity=perplexity, mode="self_search")

        edge_indices = torch.as_tensor(edge_indices, device=device, dtype=torch.long)
        edge_dist = torch.as_tensor(edge_dist, device=device, dtype=torch.float32)
        idx_tab_p = torch.as_tensor(idx_tab_p, device=device, dtype=torch.long)
        normed_coords = torch.tensor(normed_coords, device=device, dtype=torch.float32)
        
        normed_values = torch.as_tensor(normed_values, device=device, dtype=torch.float32)
        node_features = self.get_node_features(normed_coords=normed_coords, normed_values=normed_values, mode=gen_feature_mode)

        feature_distances = self.feature_distances(indices=idx_tab_p, values=normed_values)
        meta['feature_distances'] = feature_distances
        input_affinities = get_output_affinities(normed_values, edge_indices)

        edge_attributes = self.get_edge_attributes(source_coords=normed_coords,
                            target_coords=normed_coords,
                            edge_indices=edge_indices,
                            edge_dist=edge_dist,
                            k=k,
                            include_feat_diff=False)

        return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, edge_dist=edge_dist, input_affinities=input_affinities, normed_coords=normed_coords, normed_values=normed_values, meta=meta)



    @track_time
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
        
        cross_graph = Data(source = in_graph,
                           target = out_graph,
                           cross = Data(
                               edges=cross_edge_attributes,
                               edge_index=cross_edge_indices
                               )
                           )
        
        
        return cross_graph
    
    
    @track_time
    def construct_output_node_features(self, cross_graph:Data, interpolate:bool=True, gen_feature_mode:str="channel_only"):
        """
        construct node features for the output graph
        
        interpolate: if true: interpolates channel values for target graph based on neighboring values from source graph
        gen_feature_mode: determines whether features contain channel values, position or both
        """
        normed_values_source=cross_graph.source.x
        # more efficient. previously: normed_coords_source=torch.tensor(cross_graph.vert_in)
        normed_coords_source=torch.as_tensor(cross_graph.source.normed_coords)
        # more efficient. previously: normed_coords_target=torch.tensor(cross_graph.vert_out)
        normed_coords_target=torch.as_tensor(cross_graph.target.normed_coords)
        cross_edge_index=cross_graph.cross.edge_index
        
        if interpolate and (gen_feature_mode != "position_only"):
            interpolated_values = self.inverse_distance_interpolation(
                normed_values_source=normed_values_source,   # (N, F)
                normed_coords_source=normed_coords_source,   # (N, D)
                normed_coords_target=normed_coords_target,   # (M, D)
                cross_edge_index=cross_edge_index)         # (2, E)
            constructed_features=self.get_node_features(normed_coords=normed_coords_target, normed_values=interpolated_values, mode=gen_feature_mode)
            
        else:
            constructed_features=self.get_node_features(normed_coords=normed_coords_target, mode=gen_feature_mode)
        
        constructed_features = constructed_features.to(torch.float32)
        
        cross_graph.target.meta['x_true'] = cross_graph.target.x.clone()
        cross_graph.target.x = constructed_features
        
        return constructed_features
    
    
    @track_time
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
            cross_graph.cross["x"] = self.construct_output_node_features(cross_graph=cross_graph, interpolate=interpolate_graph_features, gen_feature_mode=gen_feature_mode)
            graph_list.append(cross_graph)
        return graph_list

