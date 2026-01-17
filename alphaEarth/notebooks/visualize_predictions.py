# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: neural-operators
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AlphaEarth Model Predictions Visualization
#
# This notebook loads a trained model and visualizes its predictions on a new sample.

# %%
import sys
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add paths
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(Path.cwd().parent / 'scripts'))

# Import utilities
from utilities import KernelNN, setup_device, get_output_affinities
from utilities_alphaearth import (
    prepare_dataframe,
    create_sample,
    get_global_bounds,
    load_geotiff_subregion,
    year_from_filename
)

print(f'Project root: {project_root}')

# %% [markdown]
# ## 1. Configuration
#
# Specify the model ID to load.

# %%
# Set model ID (with timestamp)
model_id = 'model1-2026-01-13_t15-08-35'  # Update this to your trained model

# Paths
trained_models_path = project_root / 'alphaEarth' / 'trained_models'
training_logs_path = project_root / 'alphaEarth' / 'training_logs'
data_path = project_root.parent / 'alphaEarth' / 'data'  # Adjust if needed

# Device setup
device = setup_device()
print(f'Using device: {device}')

# %% [markdown]
# ## 2. Load Model and Configuration

# %%
# Load full configuration (includes model, dataset, and training configs)
config_path = training_logs_path / 'configs' / f'{model_id}.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Load loss records
loss_df = pd.read_csv(training_logs_path / 'loss_records' / f'{model_id}.csv')

print('Configuration:')
for key, value in config.items():
    print(f'  {key}: {value}')

print(f'\Training completed {len(loss_df)} epochs')
print(f'Final training loss: {loss_df["loss"].iloc[-1]:.4f}')
print(f'Final validation loss: {loss_df["val_loss"].iloc[-1]:.4f}')

# %%
# Initialize model with configuration
model = KernelNN(
    width=config['width'],
    ker_width=config['ker_width'],
    depth=config['depth'],
    ker_in=config['edge_attr'],
    in_width=config['node_features'],
    out_width=config['out_width'],
).to(device)

# Load trained weights
model_path = trained_models_path / f'{model_id}.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'Model loaded: {model_id}')
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

# %% [markdown]
# ## 3. Create New Sample
#
# Use the same functionality as dataset creation to generate a single sample.

# %%
# Choose a year and file (use test year for out-of-sample prediction)
test_year = 2019  # First test year
tif_path = data_path / f'sat_emb_{test_year}.tif'

if not tif_path.exists():
    print(f'Warning: File not found: {tif_path}')
    print('Available files:')
    for f in sorted(data_path.glob('sat_emb_*.tif')):
        print(f'  {f.name}')
else:
    print(f'Using data file: {tif_path.name}')
    
    # Define bounding box (use config bounds)
    bbox = (
        config['lon_min'],
        config['lat_min'],
        config['lon_max'],
        config['lat_max']
    )
    
    print(f'Bounding box: lat [{bbox[1]}, {bbox[3]}], lon [{bbox[0]}, {bbox[2]}]')
    
    # Load and prepare data
    print('Loading region data...')
    df = prepare_dataframe(str(tif_path), bbox)
    print(f'Loaded {len(df)} grid points')
    
    # Create graph sample using same parameters as training
    print('Creating graph sample...')
    sample = create_sample(
        df=df,
        year=test_year,
        n_points_min= 100000, #config['n_points_min'],
        n_points_max= 100000, #config['n_points_max'],
        k=config['k'],
        perplexity=config['perplexity'],
        lifting_bool=config['lifting_bool'],
        device=device
    )
    
    print(f'Sample created:')
    print(f'  Nodes: {sample.x.shape[0]}')
    print(f'  Edges: {sample.edge_index.shape[1]}')
    print(f'  Node features: {sample.x.shape[1]}')
    print(f'  Edge features: {sample.edge_attr.shape[1]}')
    print(f'  Year: {sample.year}')

# %% [markdown]
# ## 4. Make Prediction
#
# Run the sample through the trained model.

# %%
# Move sample to device
sample = sample.to(device)

# Make prediction
with torch.no_grad():
    output = model(sample)
    predicted_affinities = get_output_affinities(output, sample.edge_index)

print(f'Prediction shape: {output.shape}')
print(f'Output affinities shape: {predicted_affinities.shape}')
print(f'Input affinities shape: {sample.input_affinities.shape}')

# %% [markdown]
# ## 5. Visualize Results
#
# Compare input and predicted affinities.

# %%
# Plot training history
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(loss_df['epoch'], loss_df['loss'], label='Training Loss', marker='o')
ax.plot(loss_df['epoch'], loss_df['val_loss'], label='Validation Loss', marker='s')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Training History - {model_id}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Compare input vs predicted affinities
input_aff = sample.input_affinities.cpu().numpy().flatten()
pred_aff = predicted_affinities.cpu().numpy().flatten()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Input affinities histogram
axes[0].hist(input_aff, bins=50, alpha=0.7, edgecolor='black')
axes[0].set_title('Input Affinities')
axes[0].set_xlabel('Affinity Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Predicted affinities histogram
axes[1].hist(pred_aff, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1].set_title('Predicted Affinities')
axes[1].set_xlabel('Affinity Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

# Scatter plot: input vs predicted
axes[2].scatter(input_aff, pred_aff, alpha=0.3, s=1)
axes[2].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
axes[2].set_title('Input vs Predicted Affinities')
axes[2].set_xlabel('Input Affinity')
axes[2].set_ylabel('Predicted Affinity')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compute correlation
correlation = np.corrcoef(input_aff, pred_aff)[0, 1]
print(f'Correlation between input and predicted affinities: {correlation:.4f}')

# %%
import numpy as np
import matplotlib.pyplot as plt

# Node locations
locations = sample.locations

fig, ax = plt.subplots(figsize=(10, 8))

# --- Plot edges FIRST (so they stay behind nodes) ---
edge_index = sample.edge_index.cpu().numpy()
num_edges = edge_index.shape[1]

for i in range(num_edges):
    src, dst = edge_index[:, i]
    ax.plot(
        [locations['lon'].iloc[src], locations['lon'].iloc[dst]],
        [locations['lat'].iloc[src], locations['lat'].iloc[dst]],
        color='black',
        alpha=0.1,      # low alpha to avoid clutter
        linewidth=0.5,
        zorder=1
    )

# --- Plot nodes ---
ax.scatter(
    locations['lon'],
    locations['lat'],
    color='tab:blue',
    s=50,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5,
    zorder=2
)

# --- Labels & styling ---
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(
    f'Graph Structure: {sample.x.shape[0]} nodes, {num_edges} edges'
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# # Napari RGB Visualization of Model Predictions
#
# Visualize the 3 output dimensions of the model's predictions as RGB channels.

# %%
A = np.array(output.cpu().numpy())
B = np.array(sample.locations)

# %%
# Per point plot

# --- Normalize colors ---
A = A.astype(float)

norms = np.linalg.norm(A, axis=1, keepdims=True)
norms[norms == 0] = 1.0
A_norm = A / norms

# Optional brightness scaling
V = 0.6
colors = np.clip(A_norm * V, 0, 1)

# --- Coordinates ---
points = np.column_stack([B[:, 0], B[:, 1]])  # (lat, lon)

# --- Napari viewer ---
viewer = napari.Viewer()

viewer.add_points(
    points,
    face_color=colors,
    size=5,
    blending='additive',
    name='nodes'
)

napari.run()

# %%
# Plot as interpolated images

import numpy as np
import napari
from scipy.interpolate import griddata

# -----------------------------
# Input data
# A: (N, 3)  -> non-normalized RGB channels
# B: (N, 2)  -> (latitude, longitude)
# -----------------------------

A = A.astype(float)
lat = B[:, 0]
lon = B[:, 1]

# -----------------------------
# Per-channel normalization
# -----------------------------

A_norm = np.zeros_like(A)

for c in range(3):
    channel = A[:, c]
    cmin = channel.min()
    cmax = channel.max()
    if cmax > cmin:
        A_norm[:, c] = (channel - cmin) / (cmax - cmin)
    else:
        A_norm[:, c] = 0.0

# Optional global brightness scaling (prevents additive washout)
V = 0.5
A_norm = np.clip(A_norm * V, 0, 1)

# -----------------------------
# Create regular lat–lon grid
# -----------------------------

grid_size = 300  # adjust resolution here

lat_grid = np.linspace(lat.min(), lat.max(), grid_size)
lon_grid = np.linspace(lon.min(), lon.max(), grid_size)

lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

points = np.column_stack([lon, lat])

# -----------------------------
# Interpolate each channel
# -----------------------------

R = griddata(points, A_norm[:, 0], (lon_mesh, lat_mesh), method='linear')
G = griddata(points, A_norm[:, 1], (lon_mesh, lat_mesh), method='linear')
B = griddata(points, A_norm[:, 2], (lon_mesh, lat_mesh), method='linear')

R[np.isnan(R)] = 0.0
G[np.isnan(G)] = 0.0
B[np.isnan(B)] = 0.0

# -----------------------------
# Stack into RGB image
# -----------------------------

rgb_image = np.stack([R, G, B], axis=-1)  # (H, W, 3)

# -----------------------------
# Napari visualization
# -----------------------------

viewer = napari.Viewer()

viewer.add_image(
    rgb_image,
    name="Interpolated RGB",
    rgb=True,
    interpolation="linear"
)

napari.run()


# %%
import numpy as np
import napari
from scipy.interpolate import griddata

# -----------------------------
# Input data
# A: (N, 3)  -> non-normalized RGB channels
# B: (N, 2)  -> (latitude, longitude)
# -----------------------------

A = A.astype(float)
lat = B[:, 0]
lon = B[:, 1]

# -----------------------------
# Per-channel normalization
# -----------------------------

A_norm = np.zeros_like(A)

for c in range(3):
    channel = A[:, c]
    cmin = channel.min()
    cmax = channel.max()
    if cmax > cmin:
        A_norm[:, c] = (channel - cmin) / (cmax - cmin)
    else:
        A_norm[:, c] = 0.0

# Optional global brightness scaling (prevents additive washout)
V = 0.5
A_norm = np.clip(A_norm * V, 0, 1)

# -----------------------------
# Create regular lat–lon grid
# -----------------------------

grid_size = 300  # adjust resolution here

lat_grid = np.linspace(lat.min(), lat.max(), grid_size)
lon_grid = np.linspace(lon.min(), lon.max(), grid_size)

lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

points = np.column_stack([lon, lat])

# -----------------------------
# Interpolate each channel
# -----------------------------

Red = griddata(points, A_norm[:, 0], (lon_mesh, lat_mesh), method='linear')
Green = griddata(points, A_norm[:, 1], (lon_mesh, lat_mesh), method='linear')
Blue = griddata(points, A_norm[:, 2], (lon_mesh, lat_mesh), method='linear')

Red[np.isnan(Red)] = 0.0
Green[np.isnan(Green)] = 0.0
Blue[np.isnan(Blue)] = 0.0

# -----------------------------
# Stack into 3 separate layers
# -----------------------------

image_stack = np.stack([Red, Green, Blue], axis=0)  # shape: (3, H, W)
colormap_names = ['red', 'green', 'blue']
opacity_per_layer = 0.75  # reasonable for additive blending

# -----------------------------
# Napari visualization
# -----------------------------

viewer = napari.Viewer()

for i in range(3):
    viewer.add_image(
        image_stack[i],
        name=f'channel_{colormap_names[i]}',
        colormap=colormap_names[i],
        blending='additive',
        opacity=opacity_per_layer,
        interpolation='cubic'
    )

napari.run()


# %%
