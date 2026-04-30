# %%
"""
Dataset Creation Script for MODIS Himalaya Data
=====================================

Creates train and test LMDB datasets from MODIS Himalaya satellite data.
Each sample uses a random number of points from the fixed region.

Graph structure:
- Single graph with k-NN connectivity
- Uses timestamps from config for data selection

Usage:
    python create_dataset.py <config_name>
    
    config_name: Name of config file in configurations/regular/dataset_configs/ (without .json extension)

Example:
    python create_dataset.py base
"""


import sys
import os
import json
import torch
import random


# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/modis_himalaya/scripts/")
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from dataset_utilities import (
    GraphLMDBWriter,
)

from utilities_modis_himalaya import MODIS_Himalaya_Graph_Generator, clear_memory_cache


config_id = sys.argv[1]


def main(config_id):
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
    
    N_min = config['N_min']
    N_max = config['N_max']
    k = config['k']
    perplexity = config['perplexity']
    sampling_method = config['sampling_method']
    sigma_method = config['sigma_method']
    bbox = config['bbox']
    
    min_file_size_mb = config.get('min_file_size_mb', 250)
    
    train_min_date = config.get('train_min_date')
    train_max_date = config.get('train_max_date')
    test_min_date = config.get('test_min_date')
    test_max_date = config.get('test_max_date')
    
    Generator = MODIS_Himalaya_Graph_Generator(config["data_folder_path"])
    
    train_timestamps = Generator.get_available_timestamps(
        min_date=train_min_date,
        max_date=train_max_date,
        min_file_size_mb=min_file_size_mb
    )
    
    test_timestamps = Generator.get_available_timestamps(
        min_date=test_min_date,
        max_date=test_max_date,
        min_file_size_mb=min_file_size_mb
    )
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    
    for timestamp in train_timestamps:
        success = Generator.load_data(timestamp=timestamp, bbox=bbox)
        if not success:
            print(f"Skipping timestamp due to load failure: {timestamp}")
            continue
        
        for n in range(config["train_samples_per_timestamp"]):
            print(f'{timestamp}: {n+1}/{config["train_samples_per_timestamp"]}', end='\r')
            N = random.randint(N_min, N_max)
            graph = Generator.sample_graph(
                N=N, 
                k=k, 
                perplexity=perplexity, 
                sampling_method=sampling_method, 
                gen_feature_mode=config["feature_mode"],
                sigma_method=sigma_method
            )
            writer.append(graph)
        print()
        clear_memory_cache()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for timestamp in test_timestamps:
        success = Generator.load_data(timestamp=timestamp, bbox=bbox)
        if not success:
            print(f"Skipping timestamp due to load failure: {timestamp}")
            continue
        
        for n in range(config["test_samples_per_timestamp"]):
            print(f'{timestamp}: {n+1}/{config["test_samples_per_timestamp"]}', end='\r')
            N = random.randint(N_min, N_max)
            graph = Generator.sample_graph(
                N=N, 
                k=k, 
                perplexity=perplexity, 
                sampling_method=sampling_method, 
                gen_feature_mode=config["feature_mode"],
                sigma_method=sigma_method
            )
            writer.append(graph)
        print()
        clear_memory_cache()
    
    # Save config alongside dataset
    config_save_path = config["dataset_save_path"]
    with open(f'{config_save_path}/{config_id}_config.json', 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main(config_id)
# %%