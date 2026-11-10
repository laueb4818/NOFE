import sys
import json
import random
import os
import argparse
import numpy as np
import pandas as pd
import torch
import time
from scipy.interpolate import griddata
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(Path.cwd().parent / 'scripts'))

# Import utilities
from utilities import KernelNN, KernelNN_subgraph, setup_device, get_output_affinities

# Device setup
device = setup_device()


def load_model(id:str, model_type:str='random_crop'):
    model_path = project_root / 'ERA5' / 'trained_models' / model_type / id
    with open(model_path, 'r') as f:
        config = json.load(f)
    model = KernelNN(
        width=config['width'],
        ker_width=config['ker_width'],
        depth=config['depth'],
        ker_in=config['edge_attr'],
        in_width=config['node_features'],
        out_width=config['out_width']
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Measure preprocessing (graph sampling) and embedding (model forward) cost"
    )
    parser.add_argument(
        "--preprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-sample graphs and time preprocessing (default: True). Disable with --no-preprocessing."
    )
    parser.add_argument(
        "--embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the model forward pass and time embedding (default: False). Disable with --no-embedding."
    )
    args = parser.parse_args()
    preprocessing = args.preprocessing
    embedding = args.embedding
    print(f'\npreprocess data: {preprocessing}\ncalculate embeddings: {embedding}\n')

    name = "computation_times"
    if preprocessing:
        name = name + "_preproc"
    if embedding:
        name = name + "_embed"
    
    if not preprocessing and not Path("graphs.pt").exists():
        print("Error: graphs.pt not found (set preprocessing to True to create it)")
        sys.exit(1)

    # choose data
    model_type = 'regular'


    # ---------------------------------------------------------------------------
    sys.path.append(f"{project_root}/ERA5/scripts/")
    from utilities_era5 import ERA5_Graph_Generator as Data_Generator


    # Paths
    trained_models_path = project_root / 'ERA5' / 'trained_models' / model_type
    training_logs_path = trained_models_path / 'logs'
    data_path = project_root / 'ERA5' / 'data'  # Adjust if needed

    # Set model ID
    model_id = 'regular_papermodel2_2026-05-04_t12-01-43'


    # ---------------------------------------------------------------------------
    if model_id.endswith('.pth'):
        model_id = model_id[:-4]

    # Load full configuration (includes model, dataset, and training configs)
    config_path = f'{training_logs_path}/configs/{model_id}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load loss records
    loss_df = pd.read_csv(f'{training_logs_path}/loss_records/{model_id}.csv')


    lat_min = config["lat_min"]
    lat_max = config["lat_max"]
    lon_min = config["lon_min"]
    lon_max = config["lon_max"]
    #N = config["N"]
    #k = config["k"]
    perplexity = config["perplexity"]
    sampling_method = config["sampling_method"]
    train_date_start_end = config["train_date_start_end"]
    test_date_start_end = config["test_date_start_end"]
    train_samples_per_date = config["train_samples_per_date"]
    test_samples_per_date = config["test_samples_per_date"]
    feature_mode = config["feature_mode"]
    vars = config["vars"]



    if embedding:
        model = KernelNN(
            width=config['width'],
            ker_width=config['ker_width'],
            depth=config['depth'],
            ker_in=config['edge_attr'],
            in_width=config['node_features'],
            out_width=config['out_width']
        ).to(device)

        # Load trained weights
        model_path = f'{trained_models_path}/{model_id}.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print(f'Model loaded: {model_id}')
        print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')




    date = "2019-01-15"

    #bbox = lon_min, lat_min, lon_max, lat_max
    bbox = -50, -50, 50, 75
    Generator = Data_Generator(data_path)
    Generator.load_data(bbox=bbox, vars=vars, filename=f'{date}.nc')
    print(f'total points available: {Generator.da.shape[1]*Generator.da.shape[2]}')





    sample_sizes = [500, 1000, 5000, 10000]
    k_values = [2, 6, 10]
    runs = 30


    k_list = []
    N_list = []
    run_list = []
    preprocessing_time_list = []
    reduction_time_list = []


    if preprocessing:
        graphs = []
    else:
        graphs = torch.load(
            "graphs.pt",
            map_location="cpu",
            weights_only=False,
        )

    # --- warmup ---
    print('warmup ...', end='\r')
    if preprocessing:
        graph = Generator.sample_graph(N=sample_sizes[0], k=k_values[0], perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=feature_mode)
    else:
        graph = graphs[0].to(device)

    if embedding:
        output = model(graph)

    print('warmup finished')

    # --- begin iteration ---
    g = 0
    for i, k in enumerate(k_values):
        for j, N in enumerate(sample_sizes):

            print(f'processing: \tk = {k}\tN = {N}')

            for r in range(runs + 1):
                # first iteration of every (k, N) cell is a discarded warmup and is never recorded
                warmup = r == 0
                # preprocessing
                if preprocessing:
                    start_preproc = time.perf_counter()
                    graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=feature_mode)
                    end_preproc = time.perf_counter()
                    preprocessing_time = end_preproc-start_preproc
                    if not warmup:
                        graphs.append(graph.detach().cpu())
                else:
                    graph = graphs[g % len(graphs)]
                    if not warmup:
                        g += 1 # iterating through graphs in the same order as they are generated
                    preprocessing_time = np.nan

                # embedding
                if embedding:
                    graph = graph.to(device)
                    start = time.perf_counter()
                    output = model(graph)
                    end = time.perf_counter()
                    reduction_time = end-start
                else:
                    reduction_time = np.nan

                if warmup:
                    continue

                k_list.append(k)
                N_list.append(N)
                run_list.append(r - 1)
                preprocessing_time_list.append(preprocessing_time)
                reduction_time_list.append(reduction_time)



            records = pd.DataFrame({'k': k_list,
                                    'N': N_list,
                                    'run': run_list,
                                    'preprocessing_time': preprocessing_time_list,
                                    'reduction_time': reduction_time_list,
                                    })


            records.to_csv(f"{name}.csv", index=False)

    if preprocessing:
        print('writing graphs to file')
        torch.save(graphs, "graphs.pt")


if __name__ == '__main__':
    main()
