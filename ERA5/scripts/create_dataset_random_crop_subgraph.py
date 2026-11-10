# %%
"""
ERA5 random-crop subgraph dataset creation.

Reads its configuration from ../configurations/random_crop_subgraph/dataset_configs/<config_name>.json and writes
the LMDB datasets to the dataset_save_path recorded in that config.

Usage:
    python create_dataset_random_crop_subgraph.py <config_name>
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


def sample_random_bbox(min_lat_span=10, min_lon_span=10):
    """
    Generate a random bounding box.
    
    Returns:
        tuple: (lon_min, lat_min, lon_max, lat_max)
    """
    lat_min = random.uniform(-90, 90 - min_lat_span)
    lat_max = random.uniform(lat_min + min_lat_span, 90)
    lon_min = random.uniform(-180, 180 - min_lon_span)
    lon_max = random.uniform(lon_min + min_lon_span, 180)
    return lon_min, lat_min, lon_max, lat_max


def sample_N(N_min, N_max):
    """
    Sample number of points uniformly within range.
    """
    return random.randint(N_min, N_max)


def main(config_id):
    """
    Generate random crop dataset with varying bounding boxes and point counts.
    """
    
    # Open and load the JSON
    with open("../configurations/random_crop_subgraph/dataset_configs/"+config_id+".json", "r") as f:
        config = json.load(f)
    
    print("Configuration:\n")
    print(*[f"{k}: {v}" for k, v in config.items()], sep="\n")
    
        
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_train.lmdb'):
        print('dataset already exists -- STOP')
        return()
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_test.lmdb'):
        print('dataset already exists -- STOP')
        return()
    
    # Load config parameters
    min_lat_span = config.get('min_lat_span', 10)
    min_lon_span = config.get('min_lon_span', 10)
    
    train_dates = pd.date_range(config["train_date_start_end"][0], config["train_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    test_dates = pd.date_range(config["test_date_start_end"][0], config["test_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    Generator = ERA5_Graph_Generator(config["data_folder_path"])

    for date in train_dates:        
        for n in range(config["train_samples_per_date"]):
            N_in = sample_N(config['N_in_min'], config['N_in_max'])
            N_out = sample_N(config['N_out_min'], config['N_out_max'])
            bbox = sample_random_bbox(min_lat_span, min_lon_span)
            Generator.load_data(bbox=bbox, vars=config['vars'], filename=f'{date}.nc')
            out_graph = Generator.sample_graph(N=N_out, k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            print(f'{date}: {n+1}/{config["train_samples_per_date"]}', end='\r')
            in_graph = Generator.sample_graph(N=N_in, k=config["k_in"], perplexity=config["perplexity"], sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph)
        print()
        clear_memory_cache()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for date in test_dates:
        
        for n in range(config["test_samples_per_date"]):
            N_in = sample_N(config['N_in_min'], config['N_in_max'])
            N_out = sample_N(config['N_out_min'], config['N_out_max'])
            bbox = sample_random_bbox(min_lat_span, min_lon_span)
            Generator.load_data(bbox=bbox, vars=config['vars'], filename=f'{date}.nc')
            out_graph = Generator.sample_graph(N=N_out, k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            print(f'{date}: {n+1}/{config["train_samples_per_date"]}', end='\r')
            in_graph = Generator.sample_graph(N=N_in, k=config["k_in"], perplexity=config["perplexity"], sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph)
        print()
        clear_memory_cache()

    # Save config alongside dataset
    config_save_path = config["dataset_save_path"]
    with open(f'{config_save_path}/{config_id}_config.json', 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main(config_id)
# %%
