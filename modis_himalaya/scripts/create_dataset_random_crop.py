# %%
"""
MODIS Himalaya random-crop dataset creation.

Reads its configuration from ../configurations/random_crop/dataset_configs/<config_name>.json and writes
the LMDB datasets to the dataset_save_path recorded in that config.

Usage:
    python create_dataset_random_crop.py <config_name>
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
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utilities import GraphLMDBWriter

from utilities_modis_himalaya import MODIS_Himalaya_Graph_Generator, clear_memory_cache, _function_timings


config_id = sys.argv[1]


def sample_random_bbox(max_row, max_col, crop_height_min, crop_height_max, crop_width_min, crop_width_max):
    """
    Generate a random bounding box in pixel coordinates.
    
    Returns:
        tuple: (row_min, col_min, row_max, col_max)
    """
    crop_h = random.randint(crop_height_min, crop_height_max)
    crop_w = random.randint(crop_width_min, crop_width_max)
    
    row_min = random.randint(0, max_row - crop_h)
    col_min = random.randint(0, max_col - crop_w)
    
    return (row_min, col_min, row_min + crop_h, col_min + crop_w)


def main(config_id):
    with open("../configurations/random_crop/dataset_configs/"+config_id+".json", "r") as f:
        config = json.load(f)
    
    print("Configuration:\n")
    print(*[f"{k}: {v}" for k, v in config.items()], sep="\n")
    
    # Resolve paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    data_folder_path = os.path.join(project_root, config["data_folder_path"])
    dataset_save_path = os.path.join(project_root, config["dataset_save_path"])
    
    print(f"\nData path: {data_folder_path}")
    print(f"Dataset save path: {dataset_save_path}")
    
    if os.path.exists(f'{dataset_save_path}/{config_id}_train.lmdb'):
        print('dataset already exists -- STOP')
        return()
    if os.path.exists(f'{dataset_save_path}/{config_id}_test.lmdb'):
        print('dataset already exists -- STOP')
        return()
    
    crop_height_min = config['crop_height_min']
    crop_height_max = config['crop_height_max']
    crop_width_min = config['crop_width_min']
    crop_width_max = config['crop_width_max']
    
    N_min = config['N_min']
    N_max = config['N_max']
    k = config['k']
    perplexity = config['perplexity']
    sampling_method = config['sampling_method']
    
    
    min_file_size_mb = config.get('min_file_size_mb', 250)
    
    train_min_date = config.get('train_min_date')
    train_max_date = config.get('train_max_date')
    test_min_date = config.get('test_min_date')
    test_max_date = config.get('test_max_date')
    
    Generator = MODIS_Himalaya_Graph_Generator(data_folder_path)
    
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
    
    # Get image dimensions from first file for bbox sampling
    if train_timestamps:
        Generator.load_data(timestamp=train_timestamps[0])
        max_row, max_col = Generator.da.shape[1], Generator.da.shape[2]
    else:
        print("No valid timestamps found -- STOP")
        return()
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{dataset_save_path}/{config_id}_train.lmdb')
    
    for timestamp in train_timestamps:
        Generator.load_data(timestamp=timestamp)
        
        # Update max dimensions in case they changed
        max_row, max_col = Generator.da.shape[1], Generator.da.shape[2]
        
        for n in range(config["train_samples_per_timestamp"]):
            print(f'{timestamp}: {n+1}/{config["train_samples_per_timestamp"]}', end='\r')
            N = random.randint(N_min, N_max)
            
            bbox = sample_random_bbox(
                max_row, max_col,
                crop_height_min, crop_height_max,
                crop_width_min, crop_width_max
            )
            
            graph = Generator.sample_graph(N=N, 
            k=k, 
            perplexity=perplexity, 
            sampling_method=sampling_method, 
            gen_feature_mode=config["feature_mode"], bbox=bbox)
            writer.append(graph)
        print()
        clear_memory_cache()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{dataset_save_path}/{config_id}_test.lmdb')
    
    for timestamp in test_timestamps:
        Generator.load_data(timestamp=timestamp)
        
        max_row, max_col = Generator.da.shape[1], Generator.da.shape[2]
        
        for n in range(config["test_samples_per_timestamp"]):
            print(f'{timestamp}: {n+1}/{config["test_samples_per_timestamp"]}', end='\r')
            N = random.randint(N_min, N_max)
            
            bbox = sample_random_bbox(
                max_row, max_col,
                crop_height_min, crop_height_max,
                crop_width_min, crop_width_max
            )
            
            graph = Generator.sample_graph(N=N, 
            k=k, 
            perplexity=perplexity, 
            sampling_method=sampling_method, 
            gen_feature_mode=config["feature_mode"], bbox=bbox)
            writer.append(graph)
        print()
        clear_memory_cache()
    
    # Save config alongside dataset
    with open(f'{dataset_save_path}/{config_id}_config.json', 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main(config_id)