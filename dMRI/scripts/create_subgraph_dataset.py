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
import pandas as pd
import random


# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/dMRI/scripts/" )
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from dataset_utilities import (
    GraphLMDBWriter,
    #GraphLMDBReader,
)

from utilities_dMRI import dMRI_Graph_Generator



config_id = sys.argv[1]

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
    Generator = dMRI_Graph_Generator(config["data_folder_path"])
    
    
    all_files = os.listdir(config["data_folder_path"])
    datafolders=[]
    for f in all_files:
        try:
            f = int(f)
        except:
            print(f'skipping file "{f}"')
        if type(f)==int:
            datafolders.append(str(f))
    print(f'available files:\n{datafolders}\n')
    random.shuffle(datafolders)
    train_folders = datafolders[:int(0.7*len(datafolders))]
    test_folders = datafolders[int(0.8*len(datafolders)):int(0.9*len(datafolders))]
    remaining_folders = datafolders[int(0.9*len(datafolders)):]
    print(f'Used for training data:\n{train_folders}\n')
    print(f'Used for validation data:\n{test_folders}\n')
    print(f'Remaining for tests:\n{remaining_folders}\n')
    
    for folder in train_folders:
        path = config['data_folder_path']+folder+"/T1w/Diffusion/"
        Generator.load_data(path = path)
        
        if config['output_sampling_method'] == "regular_subset":
            print('\n\n\nWARNING: the output_sampling_method "regular_subset" is not implemented yet!!!\n\n\nuse random for now')
            break
            #out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
        for n in range(config["train_samples_per_file"]):
            
            # if output graph is not reused, it has to be created inside the loop
            if config['output_sampling_method'] != "regular_subset":
                out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            
            print(f'{folder}: {n+1}/{config["train_samples_per_file"]}', end='\r')
            in_graph = Generator.sample_graph(N=config["N_in"], k=config["k_in"], perplexity=config["perplexity"], sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph)
        print()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for folder in test_folders:
        path = config['data_folder_path']+folder+"/T1w/Diffusion/"
        Generator.load_data(path=path)
        
        if config['output_sampling_method'] == "regular_subset":
            print('\n\n\nWARNING: the output_sampling_method "regular_subset" is not implemented yet!!!\n\n\nuse random for now')
            break
            #out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
        for n in range(config["test_samples_per_file"]):
            
            # if output graph is not reused, it has to be created inside the loop
            if config['output_sampling_method'] != "regular_subset":
                out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            
            print(f'{folder}: {n+1}/{config["test_samples_per_file"]}', end='\r')
            in_graph = Generator.sample_graph(N=config["N_in"], k=config["k_in"], perplexity=config["perplexity"], sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph)
        print()



if __name__ == "__main__":
    main(config_id)
# %%
