# torch, faiss (k-NN/perplexity graph construction) and PHATE's numba-based internals
# all get loaded into one process here; with unconstrained thread pools that combination
# makes macOS abort with a duplicate-OpenMP-runtime error (kernel crash / segfault).
# Capping thread counts before any of those libraries import, plus the (unsafe but
# standard) escape hatch as a second line of defense, avoids it.
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import sys
import json
import random
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
import phate
import anndata as ad
import concord as ccd
import synthia as syn

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

if __name__ == "__main__":
    # No trained-model weights available -- config only, NOFE is not computed in this script.
    config = {
        'model_config_path': 'ERA5/configurations/regular/model_configs/papermodel2.json',
        'dataset_id': 'dataset_reg_fixcrop_median_woPos_tkernel',
        'dataset_path': 'ERA5/datasets/regular',
        'learning_rate': 1e-05,
        'scheduler_step': 10,
        'scheduler_gamma': 0.5,
        'epochs': 25,
        'random_seed': 42,
        'use_gradient_clipping': False,
        'track_gradients': True,
        'checkpoint_every': 5,
        'width': 64,
        'ker_width': 16,
        'depth': 3,
        'node_features': 11,
        'edge_attr': 7,
        'out_width': 3,
        'data_folder_path': '../data/',
        'dataset_save_path': '../datasets/regular/',
        'lat_min': 30,
        'lat_max': 75,
        'lon_min': -12,
        'lon_max': 33,
        'N_min': 500,
        'N_max': 10000,
        'k': 10,
        'perplexity': 10,
        'sampling_method': 'random',
        'train_date_start_end': ['2018-01-01', '2018-12-31'],
        'test_date_start_end': ['2019-01-01', '2019-01-31'],
        'train_samples_per_date': 10,
        'test_samples_per_date': 5,
        'feature_mode': 'channel_only',
        'sigma_method': 'global_median',
        'vars': ['w', 'v', 'z', 'q', 'r', 't', 'd', 'o3', 'pv', 'u', 'vo'],
    }

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

    
    #stitching_errors_nofe = []
    #stitching_errors_pca = []
    #stitching_errors_tsne = []
    #stitching_errors_umap = []
    stitching_errors_phate = []
    stitching_errors_spatialpca = []
    stitching_errors_concord = []
    stitching_errors_fpca = []

    methods=['phate', 'spatialpca', 'concord', 'fpca']

    # Start the spatialPCA R process ONCE, outside the loop -- library(SpatialPCA) (Seurat,
    # SPARK, etc.) is the slow part to load, and conda run's own activation overhead also
    # used to get paid on every single call. It now stays alive and answers one request per
    # line over stdin/stdout for the whole loop below.
    spdir = f'{project_root}/ERA5/spatialpca'
    # Unique per process so concurrently running scripts/notebooks don't collide on the
    # same buffer filenames (a fixed call_id caused FileNotFoundError / silent data
    # corruption when two spatialPCA calls raced on the same X.csv/coords.csv paths).
    spatialpca_call_id = f'spatialpca_buffer_{os.getpid()}'
    spatialpca_proc = subprocess.Popen(
        ['conda', 'run', '-n', 'spatialpca', '--no-capture-output', 'Rscript', 'execute_spatialpca_server.R'],
        cwd=spdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    ready = spatialpca_proc.stdout.readline().strip()
    if ready != 'READY':
        raise RuntimeError(f'spatialPCA server failed to start: {ready!r}\n{spatialpca_proc.stderr.read()}')
    print('spatialPCA server ready.')


    def spatialpca_embed(X, coords, call_id, n_components=3):
        np.savetxt(f'{spdir}/{call_id}_X.csv', X, delimiter=',')
        np.savetxt(f'{spdir}/{call_id}_coords.csv', coords, delimiter=',')

        spatialpca_proc.stdin.write(f'{call_id} {n_components}\n')
        spatialpca_proc.stdin.flush()
        response = spatialpca_proc.stdout.readline().strip()

        os.remove(f'{spdir}/{call_id}_X.csv')
        os.remove(f'{spdir}/{call_id}_coords.csv')

        if response != 'OK':
            raise RuntimeError(f'spatialPCA failed for {call_id}: {response}')

        Y = np.loadtxt(f'{spdir}/res_{call_id}.csv', delimiter=',', skiprows=1)
        os.remove(f'{spdir}/res_{call_id}.csv')
        return Y


    def concord_embed(X, n_components=3):
        adata = ad.AnnData(X=X.astype(np.float32))
        cur_ccd = ccd.Concord(
            adata=adata, input_feature=list(adata.var_names), device=device,
            latent_dim=n_components, verbose=False,
            # preload_dense=True: concord-sc has a bug where the num_workers kwarg is
            # accepted but never forwarded to its internal DataLoaderManager, so passing
            # num_workers=0 alone does NOT stop it from spawning ~4 worker subprocesses per
            # DataLoader. preload_dense=True hits a different code path in
            # concord/model/dataloader.py that correctly forces num_workers to 0. Without
            # this, every worker spawn re-imports torch/anndata/concord from scratch (this
            # script runs as a plain .py file, so macOS's 'spawn' start method re-executes
            # it as __main__ in each worker), which made a single date take hours instead
            # of seconds.
            preload_dense=True,
        )
        cur_ccd.fit_transform(output_key='Concord')
        return adata.obsm['Concord']


    def fpca_embed(X, n_components=3):
        generator = syn.FPCADataGenerator()
        generator.fit(X.astype(np.float64), n_fpca_components=n_components)
        return generator.eig_scores


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
        #pred_nofe_list = []
        #pred_pca_list = []
        #pred_tsne_list = []
        #pred_umap_list = []
        pred_phate_list = []
        pred_spatialpca_list = []
        pred_concord_list = []
        pred_fpca_list = []

        for sample in graphs:
            X = sample.normed_values.detach().cpu().numpy()

            """
            # NOFE
            print('NOFE ...', end='\r')
            pred_nofe = model(sample).detach().cpu().numpy()
            pred_nofe_list.append(pred_nofe)

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
            """

            # PHATE
            print('PHATE ...', end='\r')
            pred_phate = phate.PHATE(
                n_components=3, knn=15, mds_solver='smacof', random_state=42, verbose=False
            ).fit_transform(X)
            pred_phate_list.append(pred_phate)

            # spatialPCA
            print('spatialPCA ...', end='\r')
            coords = sample.normed_coords.detach().cpu().numpy()
            pred_spatialpca = spatialpca_embed(X, coords, spatialpca_call_id, n_components=3)
            pred_spatialpca_list.append(pred_spatialpca)

            # Concord
            print('Concord ...', end='\r')
            pred_concord = concord_embed(X, n_components=3)
            pred_concord_list.append(pred_concord)

            # functionalPCA
            print('fPCA ...', end='\r')
            pred_fpca = fpca_embed(X, n_components=3)
            pred_fpca_list.append(pred_fpca)

        # ==================== get indices ====================
        idx12 = np.where(graphs[0].meta.gps[:,1] > (bbox1[2]-edge_margin))[0]
        idx21 = np.where(graphs[1].meta.gps[:,1] < (bbox2[0]+edge_margin))[0]

        idx23 = np.where(graphs[1].meta.gps[:,1] < (bbox2[2]+edge_margin))[0]
        idx32 = np.where(graphs[2].meta.gps[:,1] > (bbox3[0]-edge_margin))[0]

        idx34 = np.where(graphs[2].meta.gps[:,1] < (bbox3[2]+edge_margin))[0]
        idx43 = np.where(graphs[3].meta.gps[:,1] > (bbox4[0]-edge_margin))[0]

        edge_idx_pairs = [(idx12, idx21), (idx23,idx32), (idx34,idx43)]
        edge_subset_pairs = [(0, 1), (1,2), (2,3)]
    
    
        for method_id in range(len(methods)):
            # set output_data to list of corresponding predictions, containing 4 individual results, one per region
            if methods[method_id] == 'phate':
                output_data = pred_phate_list
            elif methods[method_id] == 'spatialpca':
                output_data = pred_spatialpca_list
            elif methods[method_id] == 'concord':
                output_data = pred_concord_list
            elif methods[method_id] == 'fpca':
                output_data = pred_fpca_list

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

            if methods[method_id] == 'phate':
                stitching_errors_phate.append(avg_normed_dist)
            elif methods[method_id] == 'spatialpca':
                stitching_errors_spatialpca.append(avg_normed_dist)
            elif methods[method_id] == 'concord':
                stitching_errors_concord.append(avg_normed_dist)
            elif methods[method_id] == 'fpca':
                stitching_errors_fpca.append(avg_normed_dist)


    spatialpca_proc.stdin.write('QUIT\n')
    spatialpca_proc.stdin.flush()
    spatialpca_proc.wait(timeout=10)

    # Convert per-date mean lists to tensors
    #nofe_means = torch.tensor(stitching_errors_nofe)
    #pca_means = torch.tensor(stitching_errors_pca)
    #tsne_means = torch.tensor(stitching_errors_tsne)
    #umap_means = torch.tensor(stitching_errors_umap)
    phate_means = torch.tensor(stitching_errors_phate)
    spatialpca_means = torch.tensor(stitching_errors_spatialpca)
    concord_means = torch.tensor(stitching_errors_concord)
    fpca_means = torch.tensor(stitching_errors_fpca)

    # Print mean ± std per method
    #print(f'nofe:\t{nofe_means.mean():.4f} ± {nofe_means.std():.4f}')
    #print(f'pca:\t{pca_means.mean():.4f} ± {pca_means.std():.4f}')
    #print(f'tsne:\t{tsne_means.mean():.4f} ± {tsne_means.std():.4f}')
    #print(f'umap:\t{umap_means.mean():.4f} ± {umap_means.std():.4f}')
    print(f'phate:\t{phate_means.mean():.4f} ± {phate_means.std():.4f}')
    print(f'spatialpca:\t{spatialpca_means.mean():.4f} ± {spatialpca_means.std():.4f}')
    print(f'concord:\t{concord_means.mean():.4f} ± {concord_means.std():.4f}')
    print(f'fpca:\t{fpca_means.mean():.4f} ± {fpca_means.std():.4f}')

    print(f'total number of samples (dates) per method: {len(phate_means)}')
