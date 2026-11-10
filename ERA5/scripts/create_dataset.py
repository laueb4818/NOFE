# %%
"""
ERA5 regular (fixed bounding box) dataset creation.

Reads its configuration from ../configurations/regular/dataset_configs/<config_name>.json and writes
the LMDB datasets to the dataset_save_path recorded in that config.

Usage:
    python create_dataset.py <config_name>
"""


import sys
import os
import json
import torch
import pandas as pd
import random


# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utilities import GraphLMDBWriter

from utilities_era5 import ERA5_Graph_Generator, clear_memory_cache



config_id = sys.argv[1]

def main(config_id): #regular method
    """
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
    
    N_min=config['N_min']
    N_max=config['N_max']
    k=config['k']
    perplexity=config['perplexity']
    sampling_method=config['sampling_method']
    
    bbox = config['lon_min'], config['lat_min'], config['lon_max'], config['lat_max']
    train_dates= pd.date_range(config["train_date_start_end"][0], config["train_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    test_dates= pd.date_range(config["test_date_start_end"][0], config["test_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    Generator = ERA5_Graph_Generator(config["data_folder_path"])
    
    for date in train_dates:
        Generator.load_data(bbox=bbox, vars=config['vars'], filename=f"{date}.nc")
        
        for n in range(config["train_samples_per_date"]):
            print(f'{date}: {n+1}/{config["train_samples_per_date"]}', end='\r')
            N = random.randint(N_min, N_max)
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"])
            writer.append(graph)
        print()
        clear_memory_cache()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for date in test_dates:
        Generator.load_data(bbox=bbox, vars=config['vars'], filename=f"{date}.nc")
        
        for n in range(config["test_samples_per_date"]):
            print(f'{date}: {n+1}/{config["test_samples_per_date"]}', end='\r')
            N = random.randint(N_min, N_max)
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"])
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
