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
sys.path.append("../../src/dataset_utilities.py")

from dataset_utilities import (
    GraphLMDBWriter,
    GraphLMDBReader,
)

from utilities_era5 import ERA5_Graph_Generator



config_id = sys.argv[1]

def main(config_id): #regular method
    """
    Expect 'config_path' as '[config_folder]/[config_id]', e.g. dataset_configs_subgraph/basic
    :param config: Description
    """
    
    # Open and load the JSON
    with open("../dataset_configs/regular/"+config_id+".json", "r") as f:
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
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    
    for year in config["train_years"]:
        file_path = os.path.join(config["data_folder_path"], "sat_emb_"+str(year)+".tif")
        Generator = ERA5_Graph_Generator(file_path) 
        Generator.load_data(bbox=config["bbox"])
        
        for n in range(config["train_samples_per_year"]):
            print(f'{year}: {n+1}/{config["train_samples_per_year"]}', end='\r')
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"])
            writer.append(graph)
        print()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for year in config["test_years"]:
        file_path = os.path.join(config["data_folder_path"], "sat_emb_"+str(year)+".tif")
        Generator = ERA5_Graph_Generator(file_path) 
        Generator.load_data(bbox=config["bbox"])
        
        for n in range(config["test_samples_per_year"]):
            print(f'{year}: {n+1}/{config["test_samples_per_year"]}', end='\r')
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"])
            writer.append(graph)
        print()


if __name__ == "__main__":
    main(config_id)