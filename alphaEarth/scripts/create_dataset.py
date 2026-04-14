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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))


from utilities_alphaEarth import (
    AlphaEarth_Graph_Generator,
    GraphLMDBWriter,
    clear_memory_cache
)


config_id = sys.argv[1]

def main(config_id): #regular method
    """
    Expect 'config_path' as '[config_folder]/[config_id]', e.g. dataset_configs_subgraph/basic
    :param config: Description
    """
    
    # Open and load the JSON
    with open("../configurations/regular/dataset_configs/"+config_id+".json", "r") as f:
        config = json.load(f)
    
    print("Configuration:\n")
    print(*[f"{k}: {v}" for k, v in config.items()], sep="\n")
    
        
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
    sigma_method=config['sigma_method']
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    
    data_path = config["data_folder_path"]  
    Generator = AlphaEarth_Graph_Generator(data_path) 
    
    for year in config["train_years"]:

        Generator.load_data(bbox=config["bbox"], filename="sat_emb_"+str(year)+".tif")
        
        for n in range(config["train_samples_per_year"]):
            print(f'{year}: {n+1}/{config["train_samples_per_year"]}', end='\r')
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"], sigma_method=sigma_method)
            writer.append(graph.cpu())
        print()
        clear_memory_cache()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for year in config["test_years"]:
        Generator.load_data(bbox=config["bbox"], filename="sat_emb_"+str(year)+".tif")

        
        for n in range(config["test_samples_per_year"]):
            print(f'{year}: {n+1}/{config["test_samples_per_year"]}', end='\r')
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"], sigma_method=sigma_method)
            writer.append(graph.cpu())
        print()
        clear_memory_cache()

    # Save config alongside dataset
    config_save_path = config["dataset_save_path"]
    with open(f'{config_save_path}/{config_id}_config.json', 'w') as f:
        json.dump(config, f, indent=4)

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