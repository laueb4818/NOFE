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

Output Data object contains:

vert_in_features: node features of input graph (= node position and/or channel values)
edges_in: edge attributes of input graph
edge_idx_in: edge indices for input graph
vert_in: node positions of input graph
vert_in_gps: input graphs gps coordinates
vert_in_values: pure channel values of input graph 
edge_dist_in: node distance per edge for input graph
vert_out_features_truth: ground truth for corresponding input node feature BUT not used because for loss, we only want channel values
edges_out: edge attributes of output graph
edge_idx_out: edge indices for output graph
vert_out: node positions of output graph
vert_out_gps: output graphs gps coordinates
vert_out_values: pure channel values of output graph (THIS IS USED FOR THE GROUND TRUTH)
edge_dist_out: node distance per edge for output graph
cross_edges: edge attributes for cross connections between input and output
cross_edge_idx: edge indices for cross connections between input and output
cross_edge_dist: node distance per edge for cross connections
sigmas: sigma values per point for output nodes
input_affinities: Ground truth data --> into loss function
year: year of origin data
vert_out_features_constructed: node features to be used for the output graph. These are either just positional encodings or conatain interpolations of input channels

"""


import sys
import os
import json
import torch


# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/alphaEarth/scripts/" )
else:
    device = torch.device("cpu")
print(f'using Device: {device}')
sys.path.append("../../src/dataset_utilities.py")


from utilities_alphaEarth import (
    AlphaEarth_Graph_Generator,
    GraphLMDBWriter,
    track_time,
    _function_timings
)


config_id = sys.argv[1]



#%%

def main(config_id): #regular method
    """
    :param config: Description
    """
    
    # Open and load the JSON
    with open("../configurations/subgraph/dataset_configs/"+config_id+".json", "r") as f:
        config = json.load(f)
    
    print("Configuration:\n")
    print(*[f"{k}: {v}" for k, v in config.items()], sep="\n")
    
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_train.lmdb'):
        print('dataset already exists -- STOP')
        return()
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_test.lmdb'):
        print('dataset already exists -- STOP')
        return()
    
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    data_path = config["data_folder_path"]
    Generator = AlphaEarth_Graph_Generator(data_path) 
    
    
    for year in config["train_years"]:
        
        Generator.load_data(bbox=config["bbox"], filename="sat_emb_"+str(year)+".tif")
        
        
        # create once beofre loop if output gaph is regular -> more efficient
        if config['output_sampling_method'] == "regular_subset":
            out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
        
        for n in range(config["train_samples_per_year"]):   
            # if output graph is not reused, it has to be created inside the loop
            if config['output_sampling_method'] != "regular_subset":
                out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            
            out_graph = out_graph.to(device)
            print(f'{year}: {n+1}/{config["train_samples_per_year"]}', end='\r')
            in_graph = Generator.sample_graph(N=config["N_in"], k=config["k_in"], perplexity=None, sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph.cpu())
        print()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for year in config["test_years"]:
        Generator.load_data(bbox=config["bbox"], filename="sat_emb_"+str(year)+".tif")
        
        if config['output_sampling_method'] == "regular_subset":
            out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
        
        for n in range(config["test_samples_per_year"]):
            # if output graph is not reused, it has to be created inside the loop
            if config['output_sampling_method'] != "regular_subset":
                out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            
            out_graph = out_graph.to(device)
            print(f'{year}: {n+1}/{config["test_samples_per_year"]}', end='\r')
            in_graph = Generator.sample_graph(N=config["N_in"], k=config["k_in"], perplexity=None, sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph.cpu())
        print()
    
    
    # --- Print execution times --- #
    print("\nTiming summary:")

    # dynamic column widths
    name_width = max(len(name) for name in _function_timings) + 2
    calls_width = max(len(str(stats["calls"])) for stats in _function_timings.values())
    total_width = max(len(f"{stats['total']:.4f}") for stats in _function_timings.values())
    avg_width = max(
        len(f"{stats['total'] / stats['calls']:.4f}")
        for stats in _function_timings.values()
    )

    # headers
    header = (
        f"{'Function':<{name_width}} | "
        f"{'Calls':>{calls_width}} | "
        f"{'Total (s)':>{total_width}} | "
        f"{'Avg (s)':>{avg_width}}"
    )
    print(header)
    print("-" * len(header))

    # rows
    for name, stats in sorted(
        _function_timings.items(),
        key=lambda x: x[1]["total"],
        reverse=True,
    ):
        total = stats["total"]
        calls = stats["calls"]
        avg = total / calls

        print(
            f"{name:<{name_width}} | "
            f"{calls:>{calls_width}d} | "
            f"{total:>{total_width}.4f} | "
            f"{avg:>{avg_width}.4f}"
        )

if __name__ == "__main__":
    main(config_id)
    


