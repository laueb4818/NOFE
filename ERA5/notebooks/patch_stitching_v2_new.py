import sys
import json
import random
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import napari
from scipy.interpolate import griddata
from pathlib import Path

from plot_utils_era5 import *

project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(Path.cwd().parent / 'scripts'))

sys.path.append(f"{project_root}src/")
# Import utilities
from utilities import KernelNN, KernelNN_subgraph, GraphLMDBReader, setup_device, get_output_affinities
from plot_utilities import plot_results_napari, plot_results_matplotlib

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

def mean_dist(x):
    N = x.shape[0]
    dists = torch.cdist(x, x)
    mean_dist = dists.triu(diagonal=1).sum() / (N * (N - 1) / 2)
    return mean_dist

def nn_dist_per_point(x):
    tree = cKDTree(x)
    distances, indices = tree.query(x, k=2)
    return distances[:, 1]

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

model, config = load_model(id='test1_model5_median_tkernel_2026-04-22_t21-22-37' , model_type='regular')

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



Generator = Data_Generator(data_path)


lon_step = (lon_max - lon_min)/4
bbox1 = lon_min, lat_min, lon_min+lon_step, lat_max
bbox2 = lon_min+lon_step, lat_min, lon_min+2*lon_step, lat_max
bbox3 = lon_min+2*lon_step, lat_min, lon_min+3*lon_step, lat_max
bbox4 = lon_min+3*lon_step, lat_min, lon_min+4*lon_step, lat_max

bboxes = [bbox1, bbox2, bbox3, bbox4]


dates = pd.date_range("2019-01-01", "2019-01-31").strftime("%Y-%m-%d").tolist()

N=1000

edge_margin = 2

    
stitching_errors_nofe = []
stitching_errors_pca = []
stitching_errors_tsne = []
stitching_errors_umap = []

methods=['nofe', 'pca', 'tsne', 'umap']


for date in dates:
    print(date)
    # ==================== Generate Data ====================
    graphs=[]
    locations_list=[]
    X_list=[]
    
    for bbox in bboxes:
        Generator.load_data(bbox=bbox, vars=vars, filename=f'{date}.nc')
        print(f'total points available: {Generator.da.shape[1]*Generator.da.shape[2]}')
        graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=feature_mode, sigma_method='ones')
        sample = graph.to(device)
        graphs.append(sample)
        locations_list.append(pd.DataFrame({"lat": sample.meta.gps[:,0], "lon": sample.meta.gps[:,1]}))
        X_list.append(sample.normed_values.detach().cpu().numpy())
        
    # ==================== Predict ====================
    pred_nofe_list = []
    pred_pca_list = []
    pred_tsne_list = []
    pred_umap_list = []

    for sample in graphs:
        # NOFE
        print('NOFE ...', end='\r')
        pred_nofe = model(sample).detach().cpu().numpy()
        pred_nofe_list.append(pred_nofe)
        
        X = sample.normed_values.detach().cpu().numpy()
        
        # PCA
        print('PCA ...', end='\r')
        pca = PCA(n_components=3)
        pred_pca = pca.fit_transform(X)
        pred_pca_list.append(pred_pca)

        # t-SNE
        print('t-SNE ...', end='\r')
        pred_tsne = TSNE(
            n_components=3,
            perplexity=30,
            learning_rate="auto",
            random_state=42
        ).fit_transform(X)
        pred_tsne_list.append(pred_tsne)

        # UMAP
        print('UMAP ...', end='\r')
        umap_id = 'umap_buffer'
        dir = f'{project_root}/ERA5/umap'
        np.save(f'{dir}/{umap_id}.npy', X)
        subprocess.run(f"cd {dir} && conda run -n umap python execute_umap.py {umap_id}", shell=True)
        pred_umap = np.load(f'{dir}/res_{umap_id}.npy')
        pred_umap_list.append(pred_umap)

        os.remove(f'{dir}/{umap_id}.npy')
        os.remove(f'{dir}/res_{umap_id}.npy')
    
    # ==================== get indices ====================
    idx12 = np.where(graphs[0].meta.gps[:,1] > (bbox1[2]-edge_margin))[0]
    idx21 = np.where(graphs[1].meta.gps[:,1] < (bbox2[0]+edge_margin))[0]

    idx23 = np.where(graphs[1].meta.gps[:,1] < (bbox2[2]+edge_margin))[0]
    idx32 = np.where(graphs[2].meta.gps[:,1] > (bbox3[0]-edge_margin))[0]

    idx34 = np.where(graphs[2].meta.gps[:,1] < (bbox3[2]+edge_margin))[0]
    idx43 = np.where(graphs[3].meta.gps[:,1] > (bbox4[0]-edge_margin))[0]

    edge_idx_pairs = [(idx12, idx21), (idx23,idx32), (idx34,idx43)]
    edge_subset_pairs = [(0, 1), (1,2), (2,3)]
    
    
    for method_id in range(4):
        # set output_data to list of corresponding predictions, containing 4 individual results, one per region
        if methods[method_id] == 'nofe':
            output_data = pred_nofe_list
        elif methods[method_id] == 'pca':
            output_data = pred_pca_list
        elif methods[method_id] == 'tsne':
            output_data = pred_tsne_list
        elif methods[method_id] == 'umap':
            output_data = pred_umap_list
            
        normed_distance_tensors = []
        
        # iterating through region borders 
        for i in range(len(edge_idx_pairs)):
            
            # ============ first from patch A to B ===================
            
            # set a, b to indices to select the two patches that we want to compare
            a = edge_subset_pairs[i][0] 
            b = edge_subset_pairs[i][1]
            # set a_idx, b_idx to list of node indices within a patch, to select nodes that are in the border region to the neighboring patch
            a_idx = edge_idx_pairs[i][0]
            b_idx = edge_idx_pairs[i][1]
            
            # select subgraph according to patch and subset of nodes within the graph according to border region
            coords_a = graphs[a].normed_coords[a_idx,:].detach().cpu().numpy()
            coords_b = graphs[b].normed_coords[b_idx,:].detach().cpu().numpy()
            
            # find nearest neighbor in patch b for each node in patch a
            tree_a = cKDTree(coords_b)
            distances_a, indices_a = tree_a.query(coords_a)
            
            # set A to actual feature values of nodes in patch A, then subset to border region
            A = torch.from_numpy(output_data[a][a_idx])
            # set B to feature values of nodes in patch B, then subset to border region, then iterate through indices according to neighborhood search
            B = torch.from_numpy(output_data[b][b_idx][indices_a])
            
            nn_dists_a = nn_dist_per_point(output_data[a][a_idx])
            nn_dists_a = torch.from_numpy(nn_dists_a)
            
            normed_dist = torch.norm(A - B, dim=1)/nn_dists_a
            normed_distance_tensors.append(normed_dist)
            
            
            
            # ============ then from patch B to A ===================
            
            # set a, b to indices to select the two patches that we want to compare
            a = edge_subset_pairs[i][0] 
            b = edge_subset_pairs[i][1]
            # set a_idx, b_idx to list of node indices within a patch, to select nodes that are in the border region to the neighboring patch
            a_idx = edge_idx_pairs[i][0]
            b_idx = edge_idx_pairs[i][1]
            
            # select subgraph according to patch and subset of nodes within the graph according to border region
            coords_a = graphs[a].normed_coords[a_idx,:].detach().cpu().numpy()
            coords_b = graphs[b].normed_coords[b_idx,:].detach().cpu().numpy()
            
            # find nearest neighbor in patch a for each node in patch b
            tree_b = cKDTree(coords_a)
            distances_b, indices_b = tree_b.query(coords_b)
            
            # set A to actual feature values of nodes in patch A, then subset to border region
            B = torch.from_numpy(output_data[b][b_idx])
            # set B to feature values of nodes in patch B, then subset to border region, then iterate through indices according to neighborhood search
            A = torch.from_numpy(output_data[a][a_idx][indices_b])
            
            nn_dists_b = nn_dist_per_point(output_data[b][b_idx])
            nn_dists_b = torch.from_numpy(nn_dists_b)
            
            normed_dist = torch.norm(B-A, dim=1)/nn_dists_b
            normed_distance_tensors.append(normed_dist)

        normed_distances = torch.hstack(normed_distance_tensors)
        avg_normed_dist = normed_distances.mean()
        #print(f'{methods[method_id]}: {avg_normed_dist}')
        
        if methods[method_id] == 'nofe':
            stitching_errors_nofe.extend(normed_distances)
        elif methods[method_id] == 'pca':
            stitching_errors_pca.extend(normed_distances)
        elif methods[method_id] == 'tsne':
            stitching_errors_tsne.extend(normed_distances)
        elif methods[method_id] == 'umap':
            stitching_errors_umap.extend(normed_distances)


nofe_errs = torch.hstack(stitching_errors_nofe)
pca_errs = torch.hstack(stitching_errors_pca)
tsne_errs = torch.hstack(stitching_errors_tsne)
umap_errs = torch.hstack(stitching_errors_umap)

print(f'nofe:\t{nofe_errs.mean()}')
print(f'pca:\t{pca_errs.mean()}')
print(f'tsne:\t{tsne_errs.mean()}')
print(f'umap:\t{umap_errs.mean()}')

print(f'total number of errors calculations per method: {umap_errs.shape}')
