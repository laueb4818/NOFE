#%%
import os
import torch
from rasterio.windows import from_bounds
import numpy as np
from pyproj import Transformer
import faiss
from torch_geometric.data import Data
from torch_scatter import scatter


# %% define graph_generator
class Graph_Generator:
    """
    This class is a parent class for the actual generator calsses.
    Actual classes will need to extend this e.g. by loading
    """
    def __init__(self, datafolder:str):
        self.directory=datafolder
        
    def load_data(self):
        # should be implemented by subclass
        pass
    
    def generate_normed_sample(self, N:int, method:str):
        # should be implemented by subclass
        pass
        
        #self.config = sample_config.copy()
        # Dynamically create attributes for easy access
        #for key, value in sample_config.items():
        #    setattr(self, key, value)


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
        # more efficient. previously: edge_indices = torch.tensor(np.vstack([source_idx, target_idx]))
        edge_indices = torch.from_numpy(np.vstack([source_idx, target_idx]))
        
        
        edge_dist = distances.reshape(-1, 1)
        
        distance_table, neighbor_table = distances, indices # renaming to match other method
        
        return distance_table, neighbor_table, edge_indices, edge_dist


    def get_edge_attributes(self, source_coords:np.ndarray=None, 
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
        # more efficient. previously: edge_attributes = torch.tensor(edge_attributes, dtype=torch.float32)
        edge_attributes = torch.as_tensor(edge_attributes, dtype=torch.float32)
        
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
        #normed_values, gps = self.generate_normed_sample(N=N, method=sampling_method) # sample subset 
        #cartesian = self.cartesian_projection(lat=gps[:, 0], lon=gps[:, 1]) # get 3 dim. cartesian coordinates for sample locations
        #normed_coords = self.norm_coords(cartesian=cartesian) # norm coordinates based on subregion
        normed_values, normed_coords, meta = self.generate_normed_sample(N=N, method=sampling_method)
        distance_table, neighbor_table, edge_indices, edge_dist = self.get_edges(source=normed_coords, target=normed_coords, k=k, mode="self_search")
        edge_attributes = self.get_edge_attributes(source_coords=normed_coords, 
                            target_coords=normed_coords, 
                            edge_indices=edge_indices, 
                            edge_dist=edge_dist,
                            distance_table=distance_table,
                            k=k,
                            include_feat_diff=False)
        node_features = self.get_node_features(normed_coords=normed_coords, normed_values=normed_values, mode=gen_feature_mode)
        sigmas = self.get_sigmas(distance_table=distance_table, perplexity=perplexity)
        input_affinities = self.get_affinities(normed_values=normed_values, edge_indices=edge_indices, sigmas=sigmas)
        return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, edge_dist=edge_dist, input_affinities=input_affinities, sigmas=sigmas, normed_coords=normed_coords, normed_values=normed_values, meta=meta)



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
        cross_edge_attributes = self.get_edge_attributes(source_coords=in_graph.normed_coords, 
                            target_coords=out_graph.normed_coords, 
                            edge_indices=cross_edge_indices, 
                            edge_dist=cross_edge_dist,
                            distance_table=cross_distance_table,
                            k=k_cross,
                            include_feat_diff=False,
                            )
        
        cross_graph = Data(source = in_graph,
                           target = out_graph,
                           cross = Data(
                               edges=cross_edge_attributes,
                               edge_idx=cross_edge_indices
                               )
                           )
        
        
        return cross_graph
    
    
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
        cross_edge_idx=cross_graph.cross.edge_idx
        
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
            in_graph = self.sample_graph(N=N_in, k=k_in, perplexity=perplexity, sampling_method=input_sampling_method)
            cross_graph = self.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=k_cross)
            cross_graph.cross["x"] = self.construct_output_node_features(cross_graph=cross_graph, interpolate=interpolate_graph_features, gen_feature_mode=gen_feature_mode)
            graph_list.append(cross_graph)
        return graph_list


# %%
