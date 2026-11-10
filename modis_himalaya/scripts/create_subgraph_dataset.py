# %%
"""
MODIS Himalaya subgraph dataset creation.

Reads its configuration from ../configurations/subgraph/dataset_configs/<config_name>.json and writes
the LMDB datasets to the dataset_save_path recorded in that config.

Usage:
    python create_subgraph_dataset.py <config_name>
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


def main(config_id):
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
        Generator.load_data(timestamp=timestamp)
        
        N_out_min = config["N_out_min"]
        N_out_max = config["N_out_max"]
        N_in_min = config["N_in_min"]
        N_in_max = config["N_in_max"]
        
        if config['output_sampling_method'] == "regular_subset":
            N_out = random.randint(N_out_min, N_out_max)
            out_graph = Generator.sample_graph(N=N_out, 
            k=config["k_out"], 
            perplexity=config["perplexity"], 
            sampling_method=config["output_sampling_method"], 
            gen_feature_mode=config["feature_mode"])
        
        for n in range(config["train_samples_per_timestamp"]):
            if config['output_sampling_method'] != "regular_subset":
                N_out = random.randint(N_out_min, N_out_max)
                out_graph = Generator.sample_graph(N=N_out, 
                k=config["k_out"], 
                perplexity=config["perplexity"], 
                sampling_method=config["output_sampling_method"], 
                gen_feature_mode=config["feature_mode"])
            
            print(f'{timestamp}: {n+1}/{config["train_samples_per_timestamp"]}', end='\r')
            N_in = random.randint(N_in_min, N_in_max)
            in_graph = Generator.sample_graph(N=N_in, 
            k=config["k_in"], 
            perplexity=config["perplexity"], 
            sampling_method=config["input_sampling_method"], 
            gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(
                cross_graph=cross_graph, 
                interpolate=config["interpolate_graph_features"], 
                gen_feature_mode=config["feature_mode"]
            )
            writer.append(cross_graph)
        print()
        clear_memory_cache()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for timestamp in test_timestamps:
        Generator.load_data(timestamp=timestamp)
        
        N_out_min = config["N_out_min"]
        N_out_max = config["N_out_max"]
        N_in_min = config["N_in_min"]
        N_in_max = config["N_in_max"]
        
        if config['output_sampling_method'] == "regular_subset":
            N_out = random.randint(N_out_min, N_out_max)
            out_graph = Generator.sample_graph(N=N_out, 
            k=config["k_out"], 
            perplexity=config["perplexity"], 
            sampling_method=config["output_sampling_method"], 
            gen_feature_mode=config["feature_mode"])
        
        for n in range(config["test_samples_per_timestamp"]):
            if config['output_sampling_method'] != "regular_subset":
                N_out = random.randint(N_out_min, N_out_max)
                out_graph = Generator.sample_graph(N=N_out, 
                k=config["k_out"], 
                perplexity=config["perplexity"], 
                sampling_method=config["output_sampling_method"], 
                gen_feature_mode=config["feature_mode"])
            
            print(f'{timestamp}: {n+1}/{config["test_samples_per_timestamp"]}', end='\r')
            N_in = random.randint(N_in_min, N_in_max)
            in_graph = Generator.sample_graph(N=N_in, 
            k=config["k_in"], 
            perplexity=config["perplexity"], 
            sampling_method=config["input_sampling_method"], 
            gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(
                cross_graph=cross_graph, 
                interpolate=config["interpolate_graph_features"], 
                gen_feature_mode=config["feature_mode"]
            )
            writer.append(cross_graph)
        print()
        clear_memory_cache()
    
    # Save config alongside dataset
    config_save_path = config["dataset_save_path"]
    with open(f'{config_save_path}/{config_id}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nTiming summary:")
    
    name_width = max(len(name) for name in _function_timings) + 2
    calls_width = max(len(str(stats["calls"])) for stats in _function_timings.values())
    total_width = max(len(f"{stats['total']:.4f}") for stats in _function_timings.values())
    avg_width = max(
        len(f"{stats['total'] / stats['calls']:.4f}")
        for stats in _function_timings.values()
    )
    
    header = (
        f"{'Function':<{name_width}} | "
        f"{'Calls':>{calls_width}} | "
        f"{'Total (s)':>{total_width}} | "
        f"{'Avg (s)':>{avg_width}}"
    )
    print(header)
    print("-" * len(header))
    
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
# %%