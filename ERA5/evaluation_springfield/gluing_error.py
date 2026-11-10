import sys
import json
import random
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time

from scipy.interpolate import griddata
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(Path.cwd().parent / 'scripts'))

sys.path.append(f"{project_root}src/")
# Import utilities
from utilities import KernelNN, KernelNN_subgraph, GraphLMDBReader, setup_device, get_output_affinities

sys.path.append(f"{project_root}/ERA5/scripts/")
from utilities_era5 import ERA5_Graph_Generator as Data_Generator
from utilities_era5 import ERA5_Graph_Generator_perSampleNorm as Data_Generator_old
data_path = project_root / 'ERA5' / 'data'

# Device setup
device = setup_device()
print(f'Using device: {device}')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import subprocess

from scipy.spatial import cKDTree

def load_model(id:str, model_type:str='random_crop'):
    model_path = project_root / 'ERA5' / 'trained_models' / model_type / f'{id}.pth'
    config_path = project_root / 'ERA5' / 'trained_models' / model_type / 'logs' / 'configs' / f'{id}.json' 
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = KernelNN(
        width=config['width'],
        ker_width=config['ker_width'],
        depth=config['depth'],
        ker_in=config['edge_attr'],
        in_width=config['node_features'],
        out_width=config['out_width']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, config

model_id = sys.argv[1]

model, config = load_model(id=model_id, model_type='regular')

lat_min = config["lat_min"]
lat_max = config["lat_max"]
lon_min = config["lon_min"]
lon_max = config["lon_max"]
#N = config["N"]
k = config["k"]
perplexity = config["perplexity"]
sampling_method = config["sampling_method"]
train_date_start_end = config["train_date_start_end"]
test_date_start_end = config["test_date_start_end"]
train_samples_per_date = config["train_samples_per_date"]
test_samples_per_date = config["test_samples_per_date"]
feature_mode = config["feature_mode"]
vars = config["vars"]


lat_range = (lat_max - lat_min)/2
lon_range = (lon_max - lon_min)/2
margin=5
side_crop=7

bbox1 = lon_min+side_crop, lat_min+side_crop, lon_max-lon_range+margin, lat_max-side_crop
bbox2 = lon_min+lon_range-margin, lat_min+side_crop, lon_max-side_crop, lat_max-side_crop
bboxes = [bbox1, bbox2]

nofe_mean_dist_per_pred = []
pca_mean_dist_per_pred = []
tsne_mean_dist_per_pred = []
umap_mean_dist_per_pred = []

dates = pd.date_range("2019-01-01", "2019-01-31").strftime("%Y-%m-%d").tolist()

for date in dates:
    print(date)
        
    #Generator = Data_Generator_old(data_path)
    Generator = Data_Generator(data_path)

    N=10000

    graphs=[]
    locations_list=[]
    X_list=[]

    for bbox in bboxes:
        Generator.load_data(bbox=bbox, vars=vars, filename=f'{date}.nc')
        graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=feature_mode)
        sample = graph.to(device)
        graphs.append(sample)
        locations_list.append(pd.DataFrame({"lat": sample.meta.gps[:,0], "lon": sample.meta.gps[:,1]}))
        X_list.append(sample.normed_values.detach().cpu().numpy())


    df1 = locations_list[0]
    df2 = locations_list[1]

    # Preserve indices as columns
    tmp1 = df1.reset_index().rename(columns={'index': 'idx1'})
    tmp2 = df2.reset_index().rename(columns={'index': 'idx2'})

    # Merge on identical lat/lon pairs
    matches = tmp1.merge(tmp2, on=['lat', 'lon'])

    i1 = matches['idx1'].to_numpy()
    i2 = matches['idx2'].to_numpy()

    pred_nofe_list = []
    pred_pca_list = []
    pred_tsne_list = []
    pred_umap_list = []

    for sample in graphs:
        # NOFE
        pred_nofe = model(sample).detach().cpu().numpy()
        pred_nofe = (pred_nofe - pred_nofe.mean(axis=0)) / pred_nofe.std(axis=0)
        pred_nofe_list.append(pred_nofe)
        
        X = sample.normed_values.detach().cpu().numpy()
        
        # PCA
        pca = PCA(n_components=3)
        pred_pca = pca.fit_transform(X)
        pred_pca = (pred_pca - pred_pca.mean(axis=0)) / pred_pca.std(axis=0)
        pred_pca_list.append(pred_pca)
        """
        # t-SNE
        pred_tsne = TSNE(
            n_components=3,
            perplexity=30,
            learning_rate="auto",
            random_state=42
        ).fit_transform(X)
        pred_tsne = (pred_tsne - pred_tsne.mean(axis=0)) / pred_tsne.std(axis=0)
        pred_tsne_list.append(pred_tsne)

        # UMAP
        umap_id = 'umap_buffer'
        dir = f'{project_root}/ERA5/umap'
        np.save(f'{dir}/{umap_id}.npy', X)
        subprocess.run(f"cd {dir} && conda run -n umap python execute_umap.py {umap_id} > /dev/null 2>&1", shell=True)
        pred_umap = np.load(f'{dir}/res_{umap_id}.npy')
        pred_umap = (pred_umap - pred_umap.mean(axis=0)) / pred_umap.std(axis=0)
        pred_umap_list.append(pred_umap)

        os.remove(f'{dir}/{umap_id}.npy')
        os.remove(f'{dir}/res_{umap_id}.npy')
        """

    nofe_mean_dist_per_pred.append(np.mean((pred_nofe_list[0][i1] - pred_nofe_list[1][i2])**2, axis=1).mean())
    pca_mean_dist_per_pred.append(np.mean((pred_pca_list[0][i1] - pred_pca_list[1][i2])**2, axis=1).mean())
    #tsne_mean_dist_per_pred.append(np.mean((pred_tsne_list[0][i1] - pred_tsne_list[1][i2])**2, axis=1).mean())
    #umap_mean_dist_per_pred.append(np.mean((pred_umap_list[0][i1] - pred_umap_list[1][i2])**2, axis=1).mean())



results = {
    "nofe": nofe_mean_dist_per_pred,
    "pca": pca_mean_dist_per_pred,
    #"tsne": tsne_mean_dist_per_pred,
    #"umap": umap_mean_dist_per_pred,
}

summary = {}

for name, values in results.items():
    arr = np.array(values)

    summary[name] = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


results_dir = Path(__file__).parent / 'gluing_errors'
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / f'{model_id}.json', 'w') as f:
    json.dump(summary, f, indent=4)

print(summary)
