import sys
import json
import random
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

dates = pd.date_range("2019-01-01", "2019-01-31").strftime("%Y-%m-%d").tolist()

#Generator = Data_Generator_old(data_path)
Generator = Data_Generator(data_path)

bbox = lon_min, lat_min, lon_max, lat_max


def pairwise_distances(X):
    # X: (N, d)
    norms = np.sum(X**2, axis=1, keepdims=True)  # (N, 1)
    
    dist2 = norms + norms.T - 2 * X @ X.T        # (N, N)
    
    # numerical stability: avoid tiny negatives
    dist2 = np.maximum(dist2, 0.0)
    
    return np.sqrt(dist2)

def distance_preservation(H, L):
    DH = pairwise_distances(H)
    DL = pairwise_distances(L)
    
    # Take upper triangle (avoid duplicates + diagonal)
    i = np.triu_indices_from(DH, k=1)
    dH = DH[i]
    dL = DL[i]
    
    # Correlation (1 = perfect preservation)
    corr = np.corrcoef(dH, dL)[0, 1]
    
    return corr

def stress(H, L):
    DH = pairwise_distances(H)
    DL = pairwise_distances(L)
    
    # Take upper triangle (avoid duplicates + diagonal)
    i = np.triu_indices_from(DH, k=1)
    dH = DH[i]
    dL = DL[i]
    
    # stress 
    stress = np.sqrt(np.sum((dH - dL)**2) / np.sum(dH**2))
    
    return stress


def edge_distances(X, edge_indices):
    # X: (N, d), edge_indices is torch_geometric's (2, E): row 0 sources, row 1 targets.
    diff = X[edge_indices[0]] - X[edge_indices[1]]
    return np.linalg.norm(diff, axis=1)

def edge_distance_preservation(H, L, edge_indices):
    dH = edge_distances(H, edge_indices)
    dL = edge_distances(L, edge_indices)
    
    # Correlation
    corr = np.corrcoef(dH, dL)[0, 1]
    return corr

def edge_stress(H, L, edge_indices):
    dH = edge_distances(H, edge_indices)
    dL = edge_distances(L, edge_indices)
    
    return np.sqrt(np.sum((dH - dL)**2) / np.sum(dH**2))

#def weighted_edge_stress(H, L, edge_indices, weights):
#    dH = edge_distances(H, edge_indices)
#    dL = edge_distances(L, edge_indices)   
#    return np.sqrt(np.sum(weights * (dH - dL)**2) / np.sum(weights * dH**2))


graphs=[]
locations_list=[]
X_list=[]

for date in dates:
    #N=random.randint(1000,10000)
    N=5000
    Generator.load_data(bbox=bbox, vars=vars, filename=f'{date}.nc')
    graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=feature_mode)
    sample = graph.to(device)
    graphs.append(sample)
    locations_list.append(pd.DataFrame({"lat": sample.meta.gps[:,0], "lon": sample.meta.gps[:,1]}))
    X_list.append(sample.normed_values.detach().cpu().numpy())
    
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

    """
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


# Initialize metric lists to accumulate across all samples
nofe_preservation = []
nofe_stress = []
nofe_edge_preservation = []
nofe_edge_stress = []

pca_preservation = []
pca_stress = []
pca_edge_preservation = []
pca_edge_stress = []

for i in range(len(graphs)):

    edge_indices = graphs[i].edge_index.detach().cpu().numpy()
    H = graphs[i].normed_values.detach().cpu().numpy()
    Lnofe = pred_nofe_list[i]
    Lpca = pred_pca_list[i]
#    Ltsne = pred_tsne_list[i]
#    Lumap = pred_umap_list[i]


    nofe_preservation.append(distance_preservation(H, Lnofe))
    nofe_stress.append(stress(H, Lnofe))
    nofe_edge_preservation.append(edge_distance_preservation(H, Lnofe, edge_indices))
    nofe_edge_stress.append(edge_stress(H, Lnofe, edge_indices))

    pca_preservation.append(distance_preservation(H, Lpca))
    pca_stress.append(stress(H, Lpca))
    pca_edge_preservation.append(edge_distance_preservation(H, Lpca, edge_indices))
    pca_edge_stress.append(edge_stress(H, Lpca, edge_indices))

    '''
    tsne_preservation = []
    tsne_stress = []
    tsne_edge_preservation = []
    tsne_edge_stress = []

    tsne_preservation.append(distance_preservation(H, Ltsne))
    tsne_stress.append(stress(H, Ltsne))
    tsne_edge_preservation.append(edge_distance_preservation(H, Ltsne, edge_indices))
    tsne_edge_stress.append(edge_stress(H, Ltsne, edge_indices))


    umap_preservation = []
    umap_stress = []
    umap_edge_preservation = []
    umap_edge_stress = []

    umap_preservation.append(distance_preservation(H, Lumap))
    umap_stress.append(distance_preservation(H, Lumap))
    umap_edge_preservation.append(edge_distance_preservation(H, Lumap, edge_indices))
    umap_edge_stress.append(edge_stress(H, Lumap, edge_indices))
    '''


#scores_nofe = [np.mean(nofe_preservation), np.mean(nofe_stress), np.mean(nofe_edge_preservation), np.mean(nofe_edge_stress)]
#scores_pca = [np.mean(pca_preservation), np.mean(pca_stress), np.mean(pca_edge_preservation), np.mean(pca_edge_stress)]
#scores_tsne = [np.mean(tsne_preservation), np.mean(tsne_stress), np.mean(tsne_edge_preservation), np.mean(tsne_edge_stress)]
#scores_umap = [np.mean(umap_preservation), np.mean(umap_stress), np.mean(umap_edge_preservation), np.mean(umap_edge_stress)]


results = {
    "nofe": {
        "preservation": {"mean": float(np.mean(nofe_preservation)), "std": float(np.std(nofe_preservation))},
        "stress": {"mean": float(np.mean(nofe_stress)), "std": float(np.std(nofe_stress))},
        "edge_preservation": {"mean": float(np.mean(nofe_edge_preservation)), "std": float(np.std(nofe_edge_preservation))},
        "edge_stress": {"mean": float(np.mean(nofe_edge_stress)), "std": float(np.std(nofe_edge_stress))},
    },
    "pca": {
        "preservation": {"mean": float(np.mean(pca_preservation)), "std": float(np.std(pca_preservation))},
        "stress": {"mean": float(np.mean(pca_stress)), "std": float(np.std(pca_stress))},
        "edge_preservation": {"mean": float(np.mean(pca_edge_preservation)), "std": float(np.std(pca_edge_preservation))},
        "edge_stress": {"mean": float(np.mean(pca_edge_stress)), "std": float(np.std(pca_edge_stress))},
    }
}

results_dir = Path(__file__).parent / 'distance_preservation_scores'
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / f'{model_id}.json', 'w') as f:
    json.dump(results, f, indent=4)


print(results)
