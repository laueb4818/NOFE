# %%
"""
dMRI regular dataset creation.

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

from utilities_dMRI import dMRI_Graph_Generator, clear_memory_cache



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
    
    N=config['N']
    k=config['k']
    perplexity=config['perplexity']
    sampling_method=config['sampling_method']
    
    
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
        if not Generator.load_data(path = path):
            print(f'Skipping folder {folder} for training set')
            continue
        
        for n in range(config["train_samples_per_file"]):
            print(f'{folder}: {n+1}/{config["train_samples_per_file"]}', end='\r')
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"])
            writer.append(graph)
        print()
        clear_memory_cache()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for folder in test_folders:
        path = config['data_folder_path']+folder+"/T1w/Diffusion/"
        if not Generator.load_data(path=path):
            print(f'Skipping folder {folder} for test set')
            continue
        
        for n in range(config["test_samples_per_file"]):
            print(f'{folder}: {n+1}/{config["test_samples_per_file"]}', end='\r')
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
