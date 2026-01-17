"""
AlphaEarth Dataset Creation Utilities
======================================

This module provides utility functions for creating graph-based datasets from AlphaEarth
satellite embedding data. It adapts ERA5 utilities for GeoTIFF format with automatic
variable inclusion.

Key differences from ERA5:
- Reads GeoTIFF files instead of NetCDF
- Automatically includes all bands (no variable selection)
- Uses year-based data files instead of date-based

Many core functions are recycled from ERA5/scripts/utilities_era5.py
"""

import json
import random
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from math import pi
import sys

import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.windows import from_bounds

# Import shared utilities from common module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from dataset_utilities import (
    setup_device,
    set_seed,
    sample_gridpoints_irregular,
    caresian_projection,
    knn_faiss,
    get_edge_indices,
    lifting,
    compute_sigmas,
    get_input_affinities,
    GraphLMDBWriter,
    GraphLMDBReader,
    load_config,
    split_train_test,
    cross_knn_faiss,
    create_fixed_grid
)

from torch_geometric.data import Data


# ============================================================================
# Data Loading (adapted for GeoTIFF)
# ============================================================================

def load_geotiff_subregion(tif_path, bbox, max_pixels=30000):
    """
    Load a subregion from a GeoTIFF file with sampling.
    
    Args:
        tif_path: Path to GeoTIFF file
        bbox: (min_lon, min_lat, max_lon, max_lat)
        max_pixels: Maximum number of pixels to load (will sample if exceeded)
        
    Returns:
        DataFrame with lat, lon, and band columns
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    with rasterio.open(tif_path) as src:
        # Get window from bounds
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        
        # Read all bands
        data = src.read(window=window)
        
        # Get coordinates for each pixel
        row_start, col_start = int(window.row_off), int(window.col_off)
        rows, cols = data.shape[1], data.shape[2]
        total_pixels = rows * cols
        
        # If too many pixels, sample them
        if total_pixels > max_pixels:
            # Sample a fraction of pixels
            sample_rate = max_pixels / total_pixels
            pixel_indices = np.random.choice(total_pixels, size=max_pixels, replace=False)
            pixel_indices.sort()
            
            # Convert flat indices to row/col
            sampled_rows = pixel_indices // cols
            sampled_cols = pixel_indices % cols
            
            # Extract only sampled bands
            data_sampled = data[:, sampled_rows, sampled_cols]
            
            # Generate coordinates for sampled pixels
            xs = sampled_cols + col_start
            ys = sampled_rows + row_start
        else:
            data_sampled = data
            xs, ys = np.meshgrid(
                np.arange(col_start, col_start + cols),
                np.arange(row_start, row_start + rows)
            )
            xs = xs.flatten()
            ys = ys.flatten()
        
        # Transform to geographic coordinates
        lons, lats = rasterio.transform.xy(src.transform, ys, xs)
        
        # Create dataframe
        df_dict = {
            'lon': lons,
            'lat': lats
        }
        
        # Add all bands
        for i, band_data in enumerate(data_sampled):
            df_dict[f'band_{i}'] = band_data.flatten()
        
        df = pd.DataFrame(df_dict)
        
        # Drop NaN values and reset index to ensure sequential indexing
        df = df.dropna().reset_index(drop=True)
        
        return df


def get_global_bounds(tif_path):
    """Get global bounds from GeoTIFF file."""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        return bounds.left, bounds.bottom, bounds.right, bounds.top


def year_from_filename(filename):
    """Extract year from filename like 'sat_emb_2018.tif'."""
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else None


# ============================================================================
# AlphaEarth-specific tensor normalization (with band columns)
# ============================================================================

def norm_tensors_from_pd(df, device):
    """Normalize tensors from pandas dataframe (adapted from ERA5)."""
    # Get all band columns
    variables = [c for c in df.columns if c.startswith('band_')]
    var_tens = torch.tensor(df.loc[:, variables].values.astype(np.float32), requires_grad=False)
    coordinate_tens = torch.tensor(df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32), requires_grad=False)
    
    # Normalize variable features
    s, m = torch.std_mean(var_tens, dim=0)
    if torch.any(s == 0):
        print('Warning: std is zero for some features.')

    var_tens = (var_tens - m) / (s + 1e-6)
    var_tens = var_tens.to(device)
    
    coordinate_tens = coordinate_tens.to(device)

    # Create node_feature tensor
    # Lifting disabled
    # if lifting_bool:
    #     node_tens = lifting(var_tens)
    # else:
    node_tens = var_tens

    return var_tens, coordinate_tens, node_tens


# AlphaEarth-specific: temporal encoding for years
def encode_year(year, device):
    """Encode year as sine/cosine."""
    # Normalize year to [0, 1] range (assuming years 2000-2030)
    year_normalized = (year - 2000) / 30.0
    angle = 2 * pi * year_normalized
    y1 = torch.sin(torch.tensor(angle, dtype=torch.float32, device=device))
    y2 = torch.cos(torch.tensor(angle, dtype=torch.float32, device=device))
    return y1, y2


def create_regular_grid_from_data(df, n_lat_points, n_lon_points):
    """
    Create a regular grid using actual data points.
    
    Divides the lat/lon bounding box into n_lat_points x n_lon_points cells,
    then selects the closest actual data point in each cell.
    
    Args:
        df: DataFrame with 'lat' and 'lon' columns
        n_lat_points: Number of grid points along latitude
        n_lon_points: Number of grid points along longitude
        
    Returns:
        Tuple of (grid_indices, grid_df) where:
            - grid_indices: Indices of selected points in original df
            - grid_df: DataFrame with the selected grid points
    """
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Get bounds
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    # Create ideal grid
    lats_ideal = np.linspace(lat_min, lat_max, n_lat_points)
    lons_ideal = np.linspace(lon_min, lon_max, n_lon_points)
    
    # For each ideal grid point, find closest actual data point
    grid_indices = []
    for lat_target in lats_ideal:
        for lon_target in lons_ideal:
            # Find closest point in data
            distances = np.sqrt(
                (df['lat'].values - lat_target)**2 + 
                (df['lon'].values - lon_target)**2
            )
            closest_idx = np.argmin(distances)
            if closest_idx not in grid_indices:
                grid_indices.append(closest_idx)
    
    grid_indices = sorted(list(set(grid_indices)))  # Remove duplicates and sort
    grid_df = df.iloc[grid_indices].reset_index(drop=True)
    
    return grid_indices, grid_df


def create_output_node_features(coordinate_tens, year, device):
    """
    Create output graph node features using only positional and temporal encoding.
    No data variables are included.
    
    Args:
        coordinate_tens: Cartesian coordinates [N, 3]
        year: Year for temporal encoding
        device: PyTorch device
        
    Returns:
        node_features: [N, F] tensor with positional and temporal encodings
    """
    # Normalize coordinates to unit length (positional encoding)
    coord_norm = torch.norm(coordinate_tens, dim=1, keepdim=True)
    coord_norm = torch.clamp(coord_norm, min=1e-8)
    pos_encoding = coordinate_tens / coord_norm  # [N, 3]
    
    # Temporal encoding
    y1, y2 = encode_year(year, device)
    time_encoding = torch.stack([y1, y2], dim=0).unsqueeze(0).repeat(coordinate_tens.shape[0], 1)  # [N, 2]
    
    # Concatenate positional and temporal encodings
    node_features = torch.cat([pos_encoding, time_encoding], dim=1)  # [N, 5]
    
    return node_features


# AlphaEarth-specific: edge attributes with year encoding
def calc_edge_attr_with_year(var_tens, coordinate_tens, edges, year):
    """Calculate edge attributes with spatial and year-based temporal encoding."""
    # Feature diff
    diff = []
    for f in range(var_tens.shape[1]):
        diff.append(var_tens[edges[:, 0], f] - var_tens[edges[:, 1], f])
    diff = torch.stack(diff)
    diff = diff.T
    
    # Normalize
    m = torch.mean(diff, dim=0)
    s = torch.std(diff, dim=0)
    diff = (diff - m) / (s + 1e-6)

    # Location main node (normalized to length one)
    loc = coordinate_tens[edges[:, 0]] / torch.norm(coordinate_tens[edges[:, 0]], dim=1).unsqueeze(1)

    # Spatial distance to reference nodes
    p1 = coordinate_tens[edges[:, 0]]
    p2 = coordinate_tens[edges[:, 1]]
    vec = p1 - p2
    dist = torch.sqrt(torch.sum((vec) ** 2, dim=1)) / 1000000  # in tsd. km
    dist = dist.unsqueeze(1)

    vec_norm = torch.norm(vec, dim=1, keepdim=True)
    vec_norm = torch.clamp(vec_norm, min=1e-8)
    vec = vec / vec_norm
    
    # Year encoding
    y1, y2 = encode_year(year, coordinate_tens.device)
    year_encoding = torch.stack([y1, y2], dim=0).unsqueeze(0).repeat(edges.shape[0], 1)

    edge_attributes = torch.cat((
        diff,  # difference in node features
        loc,  # location of main node
        vec,  # normalized vector to reference node
        dist,  # distance to reference node in tsd. km
        year_encoding
    ), dim=1)

    return edge_attributes


# ============================================================================
# High-level dataset creation functions
# ============================================================================

def prepare_dataframe(
    tif_path: str,
    bbox: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Load and prepare AlphaEarth data for a given bounding box.
    
    Args:
        tif_path: Path to GeoTIFF file
        bbox: (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        DataFrame with Cartesian projection and sequential index
    """
    df = load_geotiff_subregion(tif_path, bbox)
    df = df.dropna().reset_index(drop=True)
    
    # Ensure numeric columns
    band_cols = [c for c in df.columns if c.startswith('band_')]
    for col in band_cols:
        df[col] = df[col].astype(float)
    
    # Apply Cartesian projection
    df = caresian_projection(df)
    
    return df


def create_sample(
    df: pd.DataFrame,
    year: int,
    n_points_min: int,
    n_points_max: int,
    k: int,
    perplexity: int,
    device: torch.device
) -> Data:
    """
    Create a single graph-structured data sample (adapted from ERA5).
    
    Similar to ERA5 but:
    - Uses year instead of date/time
    - All bands are automatically included
    """
    # Random point sampling
    n_points = random.randint(n_points_min, n_points_max)
    if n_points < len(df):
        df_sample = sample_gridpoints_irregular(df, n_points)
    else:
        df_sample = df
    
    # Build k-NN graph in Cartesian space
    distances, indices = knn_faiss(
        df_sample.loc[:, ['x_coord', 'y_coord', 'z_coord']],
        k=k
    )
    edges = get_edge_indices(df_sample, indices)
    
    # Normalize tensors
    var_tens, coordinate_tens, node_tens = norm_tensors_from_pd(
        df_sample,
        device=device
    )
    
    # Edge attributes with year encoding
    edge_attributes = calc_edge_attr_with_year(
        var_tens,
        coordinate_tens,
        edges,
        year
    )
    
    edge_index = torch.from_numpy(edges.T).detach()
    
    # Compute adaptive affinities
    sigmas = compute_sigmas(
        node_tens,
        edge_index,
        perplexity=perplexity,
        tol=1e-5,
        max_iter=50
    )
    
    # Create Data object
    data_sample = Data(
        x=node_tens.detach(),
        edge_index=edge_index,
        edge_attr=edge_attributes.detach(),
        locations=df_sample.loc[:, ['lat', 'lon']],
        input_affinities=get_input_affinities(var_tens, edges, sigmas=sigmas).detach(),
        year=year
    ).cpu()
    
    return data_sample


def create_lmdb_dataset(
    config: Dict[str, Any],
    years: list,
    output_path: str,
    device: torch.device,
    data_path: str = "/Volumes/T7-storage/alphaEarth/data",
    samples_per_year: int = None
) -> None:
    """Create LMDB dataset from AlphaEarth GeoTIFF files (adapted from ERA5).
    
    Args:
        config: Configuration dictionary
        years: List of years to process
        output_path: Output path for LMDB dataset
        device: Torch device
        data_path: Path to data directory
        samples_per_year: Override samples_per_year from config (used for manual split)
    """
    # Check if dataset already exists
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset already exists at: {output_path}")
        print(f"Please delete or rename the existing dataset before creating a new one.")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    k = config['k']
    perplexity = config['perplexity']
    n_points_min = config['n_points_min']
    n_points_max = config['n_points_max']
    bbox_size_deg = config.get('bbox_size_deg', None)
    
    # Use provided samples_per_year or get from config
    if samples_per_year is None:
        samples_per_year = config.get('samples_per_year')
    
    if samples_per_year is None:
        raise ValueError("samples_per_year must be provided either in config or as parameter")
    
    # Get lat/lon bounds from config
    lat_min = config.get('lat_min', None)
    lat_max = config.get('lat_max', None)
    lon_min = config.get('lon_min', None)
    lon_max = config.get('lon_max', None)
    cache_region_data = config.get('cache_region_data', False)
    
    writer = GraphLMDBWriter(output_path)
    
    print(f"Creating dataset: {output_path}")
    print(f"Processing {len(years)} years with {samples_per_year} samples per year")
    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        print(f"Region bounds: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    if bbox_size_deg is not None:
        print(f"Random sub-regions: {bbox_size_deg}° × {bbox_size_deg}°")
    else:
        print(f"Using full region for each sample (no random sub-sampling)")
    if cache_region_data:
        print(f"Region caching: enabled (load once per year)")
    else:
        print(f"Region caching: disabled (reload per sample)")
    print(f"Using device: {device}\n")
    
    try:
        for year_idx, year in enumerate(years):
            tif_path = os.path.join(data_path, f"sat_emb_{year}.tif")
            
            if not os.path.exists(tif_path):
                print(f"Warning: File not found: {tif_path}, skipping...")
                continue
            
            print(f"[{year_idx + 1}/{len(years)}] {year}: ", end='')
            
            # Determine sampling bounds
            if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
                # Use config bounds
                sampling_bounds = (lon_min, lat_min, lon_max, lat_max)
            else:
                # Use global file bounds
                sampling_bounds = get_global_bounds(tif_path)
            
            # Load region data once if caching is enabled
            cached_df = None
            if cache_region_data:
                cached_df = prepare_dataframe(tif_path, sampling_bounds)
            
            # Create output grid from first year's data if not yet created
            if not output_grid_created and cached_df is not None:
                grid_indices, grid_df_temp = create_regular_grid_from_data(cached_df, n_points_lat_out, n_points_lon_out)
                grid_df = cached_df.iloc[grid_indices].reset_index(drop=True)
                output_grid_created = True
                print(f"Output grid created with {len(grid_df)} points\n")
            
            for sample_idx in range(samples_per_year):
                print(f"\r[{year_idx + 1}/{len(years)}] {year}: {sample_idx + 1}/{samples_per_year}", end='')
                
                if cache_region_data:
                    # Use cached data and sample from it
                    df = cached_df.copy()
                else:
                    # Generate bbox for this sample
                    min_lon_g, min_lat_g, max_lon_g, max_lat_g = sampling_bounds
                    
                    if bbox_size_deg is not None:
                        # Random sub-region sampling
                        margin = 0.01
                        min_lon_g += margin
                        min_lat_g += margin
                        max_lon_g -= margin
                        max_lat_g -= margin
                        
                        # Ensure bbox fits within bounds
                        if max_lon_g - min_lon_g < bbox_size_deg or max_lat_g - min_lat_g < bbox_size_deg:
                            print(f"\nWarning: bbox_size_deg ({bbox_size_deg}) too large for region, skipping...")
                            continue
                        
                        min_lon = random.uniform(min_lon_g, max_lon_g - bbox_size_deg)
                        min_lat = random.uniform(min_lat_g, max_lat_g - bbox_size_deg)
                        bbox = (min_lon, min_lat, min_lon + bbox_size_deg, min_lat + bbox_size_deg)
                    else:
                        # Use full region for each sample
                        bbox = (min_lon_g, min_lat_g, max_lon_g, max_lat_g)
                    
                    # Load and prepare data
                    df = prepare_dataframe(tif_path, bbox)
                
                # Create sample
                data_sample = create_sample(
                    df=df,
                    year=year,
                    n_points_min=n_points_min,
                    n_points_max=n_points_max,
                    k=k,
                    perplexity=perplexity,
                    lifting_bool=lifting_bool,
                    device=device
                )
                
                writer.append(data_sample)
            
            print(f"\r[{year_idx + 1}/{len(years)}] {year}: {samples_per_year}/{samples_per_year} ✓")
        
    finally:
        writer.close()
    
    print(f"\n✓ Dataset creation complete: {output_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary (adapted from ERA5)."""
    required_fields = [
        'k', 'perplexity', 'n_points_min', 'n_points_max',
        'lifting_bool'
    ]
    
    # Check if using manual split or automatic split
    has_manual_split = 'train_years' in config or 'test_years' in config
    has_years_list = 'years' in config
    
    if not has_manual_split and not has_years_list:
        raise KeyError("Must provide either 'years' (for automatic split) or 'train_years'/'test_years' (for manual split)")
    
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Required configuration field missing: {field}")
    
    # Validate automatic split case
    if has_years_list and not has_manual_split:
        if not config['years']:
            raise ValueError("years list cannot be empty")
        if 'samples_per_year' not in config:
            raise KeyError("samples_per_year required when using automatic split")
        if config['samples_per_year'] <= 0:
            raise ValueError(f"samples_per_year must be positive")
    
    # Validate manual split case
    if has_manual_split:
        if 'train_years' not in config or 'test_years' not in config:
            raise KeyError("Both train_years and test_years required for manual split")
        if 'train_samples_per_year' not in config or 'test_samples_per_year' not in config:
            raise KeyError("Both train_samples_per_year and test_samples_per_year required for manual split")
        if config['train_samples_per_year'] <= 0:
            raise ValueError(f"train_samples_per_year must be positive")
        if config['test_samples_per_year'] <= 0:
            raise ValueError(f"test_samples_per_year must be positive")
    
    if config['k'] <= 0:
        raise ValueError(f"k must be positive, got {config['k']}")
    if config['perplexity'] <= 0:
        raise ValueError(f"perplexity must be positive, got {config['perplexity']}")
    if config['n_points_min'] <= 0:
        raise ValueError(f"n_points_min must be positive, got {config['n_points_min']}")
    if config['n_points_max'] < config['n_points_min']:
        raise ValueError(f"n_points_max must be >= n_points_min")
    
    # Validate lat/lon bounds if provided
    has_lat_bounds = 'lat_min' in config or 'lat_max' in config
    has_lon_bounds = 'lon_min' in config or 'lon_max' in config
    
    if has_lat_bounds and not ('lat_min' in config and 'lat_max' in config):
        raise ValueError("Both lat_min and lat_max must be specified together")
    if has_lon_bounds and not ('lon_min' in config and 'lon_max' in config):
        raise ValueError("Both lon_min and lon_max must be specified together")
    
    if 'lat_min' in config and 'lat_max' in config:
        if config['lat_min'] >= config['lat_max']:
            raise ValueError(f"lat_min ({config['lat_min']}) must be < lat_max ({config['lat_max']})")
        if config['lat_min'] < -90 or config['lat_max'] > 90:
            raise ValueError(f"Latitude bounds must be in [-90, 90]")
    
    if 'lon_min' in config and 'lon_max' in config:
        if config['lon_min'] >= config['lon_max']:
            raise ValueError(f"lon_min ({config['lon_min']}) must be < lon_max ({config['lon_max']})")
        if config['lon_min'] < -180 or config['lon_max'] > 180:
            raise ValueError(f"Longitude bounds must be in [-180, 180]")
    
    # Validate cache_region_data if provided
    if 'cache_region_data' in config:
        if not isinstance(config['cache_region_data'], bool):
            raise ValueError(f"cache_region_data must be a boolean, got {type(config['cache_region_data']).__name__}")


def print_dataset_summary(
    config_name: str,
    train_years: list,
    test_years: list,
    config: Dict[str, Any]
) -> None:
    """Print dataset creation summary (from ERA5)."""
    total_years = len(train_years) + len(test_years)
    
    # Determine if using manual or automatic split
    has_manual_split = 'train_years' in config and 'test_years' in config
    
    if has_manual_split:
        train_samples = config['train_samples_per_year']
        test_samples = config['test_samples_per_year']
    else:
        train_samples = config.get('samples_per_year')
        test_samples = config.get('samples_per_year')
    
    print("=" * 60)
    print(f"Dataset Configuration: {config_name}")
    print("=" * 60)
    print(f"Total years:          {total_years}")
    print(f"  Training years:     {len(train_years)}")
    print(f"  Test years:         {len(test_years)}")
    if has_manual_split:
        print(f"Samples per year:     train={train_samples}, test={test_samples}")
        print(f"  Total train samples: {len(train_years) * train_samples}")
        print(f"  Total test samples:  {len(test_years) * test_samples}")
    else:
        print(f"Samples per year:     {train_samples}")
        print(f"  Total train samples: {len(train_years) * train_samples}")
        print(f"  Total test samples:  {len(test_years) * test_samples}")
    print(f"\nGraph parameters:")
    print(f"  k (neighbors):      {config['k']}")
    print(f"  Perplexity:         {config['perplexity']}")
    print(f"  Points per sample:  {config['n_points_min']}-{config['n_points_max']}")
    
    bbox_size = config.get('bbox_size_deg', None)
    if bbox_size is not None:
        print(f"  Bbox size (deg):    {bbox_size}")
    else:
        print(f"  Bbox size:          Full region (no sub-sampling)")
    
    print(f"  Lifting enabled:    {config['lifting_bool']}")
    
    if 'lat_min' in config and 'lat_max' in config and 'lon_min' in config and 'lon_max' in config:
        print(f"\nRegion bounds:")
        print(f"  Latitude:           [{config['lat_min']}, {config['lat_max']}]")
        print(f"  Longitude:          [{config['lon_min']}, {config['lon_max']}]")
    
    print(f"\nYears: {', '.join(map(str, train_years + test_years))}")
    print("=" * 60 + "\n")


# ============================================================================
# Subgraph Dataset Creation (v2)
# ============================================================================

def calc_edge_attr_subgraph(coordinate_tens_in, coordinate_tens_out, cross_edges_np):
    """
    Calculate cross-edge attributes for subgraph datasets.
    
    Args:
        coordinate_tens_in: Input node coordinates [N_in, 3]
        coordinate_tens_out: Output node coordinates [N_out, 3]
        cross_edges_np: Edge array [E, 2] where [:, 0] = input (source), [:, 1] = output (target)
    
    Edge direction: A→B (input to output)
    Attributes: positional encoding, normalized direction vector, distance
    """
    # Source is input (A), target is output (B)
    pos_in = coordinate_tens_in[cross_edges_np[:, 0]]   # source positions
    pos_out = coordinate_tens_out[cross_edges_np[:, 1]] # target positions
    
    # Position encoding: normalized position of source (input) nodes
    pos_norm = torch.norm(pos_in, dim=1, keepdim=True)
    pos_norm = torch.clamp(pos_norm, min=1e-8)
    pos_encoding = pos_in / pos_norm
    
    # Direction vector: target - source (output - input), normalized
    direction = pos_out - pos_in
    dir_norm = torch.norm(direction, dim=1, keepdim=True)
    dir_norm = torch.clamp(dir_norm, min=1e-8)
    direction_normalized = direction / dir_norm
    
    # Distance and exponential weight
    distance = torch.norm(direction, dim=1).unsqueeze(1)
    distance_weight = torch.exp(-distance)
    
    edge_attributes = torch.cat([
        pos_encoding,           # [E, 3] - source position
        direction_normalized,   # [E, 3] - direction to target
        distance,              # [E, 1]
        distance_weight        # [E, 1]
    ], dim=1)
    
    return edge_attributes


def create_subgraph_sample(
    df_full: pd.DataFrame,
    year: int,
    n_points_input_min: int,
    n_points_input_max: int,
    grid_df: pd.DataFrame,
    k_in: int,
    k_out: int,
    k_cross: int,
    perplexity: int,
    device: torch.device,
    filename: str = None,
    cached_output: Optional[dict] = None
) -> tuple:
    """
    Create a single subgraph sample with fixed output grid and random input sampling.
    """
    # Output graph: use the regular grid (same for all samples)
    if cached_output is None:
        # Use the pre-defined grid
        df_output = grid_df.copy()
        
        # Filter out any rows with NaN in coordinates
        coord_cols = ['x_coord', 'y_coord', 'z_coord']
        df_output = df_output.dropna(subset=coord_cols).reset_index(drop=True)
        
        if len(df_output) == 0:
            raise ValueError("No valid coordinates in output grid after filtering NaN")
        
        # Raw coordinates for k-NN construction
        output_coordinates_raw = torch.tensor(
            df_output.loc[:, coord_cols].values.astype(np.float32),
            requires_grad=False,
            device=device
        )
        
        # Normalized coordinates (unit sphere) for model input
        output_coordinates = output_coordinates_raw / torch.norm(output_coordinates_raw, dim=1, keepdim=True)
        
        var_tens_out, coordinate_tens_out, node_tens_out = norm_tensors_from_pd(
            df_output,
            device=device
        )
        
        # Create output node features using only positional and temporal encoding
        node_tens_out = create_output_node_features(
            coordinate_tens_out,
            year,
            device
        )
        
        # Build output graph edges (use raw coordinates for k-NN)
        # Ensure k is valid
        k_out_actual = min(k_out, len(df_output) - 1)
        if k_out_actual < 1:
            raise ValueError(f"Not enough nodes for k-NN: got {len(df_output)} nodes, need at least 2")
        
        distances_out, edge_idx_out_out = knn_faiss(
            output_coordinates_raw.detach().cpu().numpy(),
            k=k_out_actual
        )
        edges_out_np = get_edge_indices(df_output, edge_idx_out_out)
        edge_idx_out_out = torch.from_numpy(edges_out_np.T).long().to(device)
        
        # Compute sigmas on output graph using var_tens_out (actual features)
        # This is needed for computing input affinities later
        sigmas = compute_sigmas(
            node_tens_out,
            edge_idx_out_out,
            perplexity=perplexity,
            tol=1e-5,
            max_iter=50
        )
        
        # Output edge attributes
        edge_attr_out = calc_edge_attr_with_year(
            var_tens_out,
            coordinate_tens_out,
            edges_out_np,
            year
        )
        
        # Cache output graph
        cached_output = {
            'output_coordinates': output_coordinates,
            'output_coordinates_raw': output_coordinates_raw,
            'var_tens_out': var_tens_out,
            'coordinate_tens_out': coordinate_tens_out,
            'node_tens_out': node_tens_out,
            'edge_idx_out_out': edge_idx_out_out,
            'edges_out_np': edges_out_np,
            'edge_attr_out': edge_attr_out,
            'sigmas': sigmas,
            'df_output': df_output  # Store filtered DataFrame to ensure GPS coordinates match
        }
    else:
        # Use cached output graph
        output_coordinates = cached_output['output_coordinates']
        output_coordinates_raw = cached_output['output_coordinates_raw']
        var_tens_out = cached_output['var_tens_out']
        coordinate_tens_out = cached_output['coordinate_tens_out']
        node_tens_out = cached_output['node_tens_out']
        edge_idx_out_out = cached_output['edge_idx_out_out']
        edges_out_np = cached_output['edges_out_np']
        edge_attr_out = cached_output['edge_attr_out']
        sigmas = cached_output['sigmas']
    
    # Input graph: random subset
    n_points_input = random.randint(n_points_input_min, n_points_input_max)
    n_points_input = min(n_points_input, len(df_full))
    
    input_indices = random.sample(range(len(df_full)), n_points_input)
    input_indices_sorted = sorted(input_indices)
    
    df_input = df_full.iloc[input_indices_sorted].reset_index(drop=True)
    
    # Filter out any rows with NaN in coordinates
    coord_cols = ['x_coord', 'y_coord', 'z_coord']
    df_input = df_input.dropna(subset=coord_cols).reset_index(drop=True)
    
    if len(df_input) < 2:
        raise ValueError(f"Not enough valid input nodes after filtering NaN: got {len(df_input)}, need at least 2")
    
    # Update n_points_input to reflect filtered size
    n_points_input = len(df_input)
    
    # Extract input coordinates from full DataFrame
    # Raw coordinates for k-NN construction
    input_coordinates_raw = torch.tensor(
        df_input.loc[:, coord_cols].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    
    # Normalized coordinates (unit sphere) for model input
    input_coordinates = input_coordinates_raw / torch.norm(input_coordinates_raw, dim=1, keepdim=True)
    
    # Extract GPS coordinates for input and output
    input_gps = torch.tensor(
        df_input.loc[:, ['lat', 'lon']].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    # Note: Use cached output coordinates if available, otherwise extract from current df_output
    # This is critical to ensure output_gps matches the actual filtered output nodes
    if cached_output is not None:
        output_gps_df = cached_output.get('df_output')
        if output_gps_df is not None:
            output_gps = torch.tensor(
                output_gps_df.loc[:, ['lat', 'lon']].values.astype(np.float32),
                requires_grad=False,
                device=device
            )
        else:
            # Fallback: use grid_df if df_output not cached
            output_gps = torch.tensor(
                grid_df.loc[:, ['lat', 'lon']].values.astype(np.float32),
                requires_grad=False,
                device=device
            )
    else:
        # First time creating output graph: df_output was already filtered above
        output_gps = torch.tensor(
            df_output.loc[:, ['lat', 'lon']].values.astype(np.float32),
            requires_grad=False,
            device=device
        )
    
    var_tens_in, coordinate_tens_in, node_tens_in = norm_tensors_from_pd(
        df_input,
        device=device
    )
    
    # Build input graph edges (use raw coordinates for k-NN)
    k_in_actual = min(k_in, n_points_input - 1)
    if k_in_actual < 1:
        raise ValueError(f"Not enough input nodes for k-NN: got {n_points_input} nodes, need at least 2")
    
    distances_in, edge_idx_in_in = knn_faiss(
        input_coordinates_raw.detach().cpu().numpy(),
        k=k_in_actual
    )
    edges_in_np = get_edge_indices(df_input, edge_idx_in_in)
    edge_idx_in_in = torch.from_numpy(edges_in_np.T).long().to(device)
    
    # Input edge attributes
    edge_attr_in = calc_edge_attr_with_year(
        var_tens_in,
        coordinate_tens_in,
        edges_in_np,
        year
    )
    
    # Cross connections: input to output (A→B)
    # For each output node, find k_cross nearest input nodes, then create edges FROM input TO output
    # Ensure k_cross doesn't exceed number of input nodes
    k_cross_actual = min(k_cross, n_points_input)
    if k_cross_actual < 1:
        raise ValueError(f"Not enough input nodes for cross connections: got {n_points_input} input nodes, need at least 1")
    
    # cross_knn_faiss returns [output_indices, input_indices] (output queries for nearest inputs)
    # We need edges FROM input TO output, so we flip the order to [input_indices, output_indices]
    edge_idx_cross_raw = cross_knn_faiss(
        input_coordinates_raw.detach(),
        output_coordinates_raw.detach(),
        k_cross_actual
    )
    # Flip to get [source=input, target=output] for A→B direction
    edge_idx_in_out = torch.stack([edge_idx_cross_raw[1], edge_idx_cross_raw[0]], dim=0).to(device)
    
    edges_cross_np = edge_idx_in_out.T.cpu().numpy()
    edge_attr_cross = calc_edge_attr_subgraph(
        coordinate_tens_in,
        coordinate_tens_out,
        edges_cross_np
    )
    
    # Verify consistency of graph sizes
    n_output = output_coordinates.shape[0]
    n_input = input_coordinates.shape[0]
    
    # Check output graph internal edges reference valid nodes
    if edge_idx_out_out.numel() > 0:
        max_out_idx = edge_idx_out_out.max().item()
        assert max_out_idx < n_output, f"Output edge index {max_out_idx} >= n_output {n_output}"
    
    # Check input graph internal edges reference valid nodes  
    if edge_idx_in_in.numel() > 0:
        max_in_idx = edge_idx_in_in.max().item()
        assert max_in_idx < n_input, f"Input edge index {max_in_idx} >= n_input {n_input}"
    
    # Check cross edges reference valid nodes (A→B: source=input, target=output)
    if edge_idx_in_out.numel() > 0:
        max_cross_in = edge_idx_in_out[0].max().item()   # source is input
        max_cross_out = edge_idx_in_out[1].max().item()  # target is output
        assert max_cross_in < n_input, f"Cross edge input (source) index {max_cross_in} >= n_input {n_input}"
        assert max_cross_out < n_output, f"Cross edge output (target) index {max_cross_out} >= n_output {n_output}"
    
    # Create Data object
    data_sample = Data(
        # Input subgraph
        vert_in=input_coordinates.detach(),
        vert_in_features=node_tens_in.detach(),
        vert_in_gps=input_gps.detach(),
        edge_idx_in=edge_idx_in_in,
        edges_in=edge_attr_in.detach(),
        
        # Output subgraph
        vert_out=output_coordinates.detach(),
        vert_out_features=node_tens_out.detach(),
        vert_out_gps=output_gps.detach(),
        edge_idx_out=edge_idx_out_out,
        edges_out=edge_attr_out.detach(),
        
        # Cross connections (A→B: source=input, target=output)
        cross_edge_idx=edge_idx_in_out,
        cross_edges=edge_attr_cross.detach(),
        
        # Additional info
        sigmas=sigmas.detach(),
        input_affinities=get_input_affinities(var_tens_out, edges_out_np, sigmas=sigmas).detach(),
        year=year,
        filename=filename
    ).cpu()
    
    return data_sample, cached_output


def create_interpolated_sample(
    df_full: pd.DataFrame,
    year: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat_out: int,
    n_lon_out: int,
    n_points_input_min: int,
    n_points_input_max: int,
    k: int,
    perplexity: int,
    device: torch.device,
    filename: str = None
) -> Data:
    """
    Create a sample with interpolated features on a regular output grid.
    
    This creates a single graph where:
    1. Output nodes are on a perfectly regular 2D grid
    2. Input nodes are randomly sampled from data
    3. Output node features are interpolated from input nodes
    4. All nodes have the same feature dimensions
    
    Args:
        df_full: Full dataframe with all data
        year: Year
        lat_min, lat_max: Latitude bounds for output grid
        lon_min, lon_max: Longitude bounds for output grid
        n_lat_out: Number of grid points along latitude
        n_lon_out: Number of grid points along longitude
        n_points_input_min: Minimum input points to sample
        n_points_input_max: Maximum input points to sample
        k: Number of neighbors for k-NN graph
        perplexity: Perplexity for affinity computation
        device: PyTorch device
        filename: Source filename
        
    Returns:
        Data object with single graph structure
    """
    from dataset_utilities import create_regular_2d_grid, interpolate_features_bilinear
    
    # Create regular output grid
    grid_df = create_regular_2d_grid(lat_min, lat_max, lon_min, lon_max, n_lat_out, n_lon_out)
    grid_df = caresian_projection(grid_df)
    
    # Sample random input points
    n_points_input = random.randint(n_points_input_min, n_points_input_max)
    n_points_input = min(n_points_input, len(df_full))
    input_indices = random.sample(range(len(df_full)), n_points_input)
    df_input = df_full.iloc[sorted(input_indices)].reset_index(drop=True)
    
    # Extract variable columns (excluding coordinates)
    var_cols = [c for c in df_input.columns if c not in ['lat', 'lon', 'x_coord', 'y_coord', 'z_coord']]
    
    # Interpolate features from input to output grid
    input_gps = df_input.loc[:, ['lat', 'lon']].values
    input_vars = df_input.loc[:, var_cols].values
    output_gps = grid_df.loc[:, ['lat', 'lon']].values
    
    output_vars = interpolate_features_bilinear(input_gps, input_vars, output_gps)
    
    # Combine input and output nodes
    combined_df = pd.concat([
        df_input,
        grid_df.assign(**{col: output_vars[:, i] for i, col in enumerate(var_cols)})
    ], ignore_index=True)
    
    # Get raw Cartesian coordinates
    coords_raw = torch.tensor(
        combined_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    
    # Normalize to unit sphere
    coords_normalized = coords_raw / torch.norm(coords_raw, dim=1, keepdim=True)
    
    # Build k-NN graph on combined nodes (using raw coords)
    distances, indices = knn_faiss(
        coords_raw.detach().cpu().numpy(),
        k=min(k, len(combined_df) - 1)
    )
    edges = get_edge_indices(combined_df, indices)
    
    # Normalize tensors
    var_tens, coordinate_tens, node_tens = norm_tensors_from_pd(
        combined_df,
        device=device
    )
    
    # Edge attributes
    edge_attributes = calc_edge_attr_with_year(
        var_tens,
        coordinate_tens,
        edges,
        year
    )
    
    edge_index = torch.from_numpy(edges.T).long().to(device)
    
    # Compute affinities
    sigmas = compute_sigmas(
        node_tens,
        edge_index,
        perplexity=perplexity,
        tol=1e-5,
        max_iter=50
    )
    
    # GPS coordinates
    gps_coords = torch.tensor(
        combined_df.loc[:, ['lat', 'lon']].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    
    # Create Data object
    data_sample = Data(
        x=coords_normalized.detach(),
        node_features=node_tens.detach(),
        gps=gps_coords.detach(),
        edge_index=edge_index,
        edge_attr=edge_attributes.detach(),
        input_affinities=get_input_affinities(var_tens, edges, sigmas=sigmas).detach(),
        year=year,
        filename=filename,
        n_input=n_points_input,
        n_output=len(grid_df)
    ).cpu()
    
    return data_sample


def create_subgraph_lmdb_dataset(
    config: Dict[str, Any],
    years: list,
    output_path: str,
    device: torch.device,
    data_path: str = "/Volumes/T7-storage/alphaEarth/data",
    samples_per_year: int = None
) -> None:
    """
    Create complete LMDB subgraph dataset for AlphaEarth.
    """
    # Check if dataset already exists
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset already exists at: {output_path}")
        print(f"Please delete or rename the existing dataset before creating a new one.")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Extract configuration
    k_in = config['k_in']
    k_out = config['k_out']
    k_cross = config['k_cross']
    perplexity = config['perplexity']
    n_points_input_min = config['n_points_input_min']
    n_points_input_max = config['n_points_input_max']
    n_points_lat_out = config['n_points_lat_out']
    n_points_lon_out = config['n_points_lon_out']
    
    # Use provided samples_per_year or get from config
    if samples_per_year is None:
        samples_per_year = config.get('samples_per_year')
    
    if samples_per_year is None:
        raise ValueError("samples_per_year must be provided either in config or as parameter")
    
    # Get lat/lon bounds from config
    lat_min = config.get('lat_min', None)
    lat_max = config.get('lat_max', None)
    lon_min = config.get('lon_min', None)
    lon_max = config.get('lon_max', None)
    cache_region_data = config.get('cache_region_data', False)
    bbox_size_deg = config.get('bbox_size_deg', None)
    
    writer = GraphLMDBWriter(output_path)
    
    print(f"Creating subgraph dataset: {output_path}")
    print(f"Processing {len(years)} years with {samples_per_year} samples per year")
    print(f"Graph structure: k_in={k_in}, k_out={k_out}, k_cross={k_cross}")
    print(f"Input nodes: {n_points_input_min}-{n_points_input_max} (random)")
    print(f"Output grid: {n_points_lat_out}x{n_points_lon_out} (fixed, reused across all samples)")
    print(f"Using device: {device}\n")
    
    # Create output grid once for entire dataset from first data file
    grid_df = None
    output_grid_created = False
    cached_output = None
    
    try:
        for year_idx, year in enumerate(years):
            tif_path = os.path.join(data_path, f"sat_emb_{year}.tif")
            
            if not os.path.exists(tif_path):
                print(f"Warning: File not found: {tif_path}, skipping...")
                continue
            
            print(f"[{year_idx + 1}/{len(years)}] {year}: ", end='')
            
            # Determine sampling bounds
            if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
                sampling_bounds = (lon_min, lat_min, lon_max, lat_max)
            else:
                sampling_bounds = get_global_bounds(tif_path)
            
            # Load region data once if caching is enabled
            cached_df = None
            if cache_region_data:
                cached_df = prepare_dataframe(tif_path, sampling_bounds)
            
            # Create output grid from first year's data if not yet created
            if not output_grid_created:
                if cache_region_data:
                    df_for_grid = cached_df
                else:
                    df_for_grid = prepare_dataframe(tif_path, sampling_bounds)
                
                # Filter out NaN values before creating grid
                df_for_grid = df_for_grid.dropna()
                
                if len(df_for_grid) == 0:
                    print(f"Warning: No valid data points after filtering NaN for year {year}, skipping...")
                    continue
                
                grid_indices, grid_df_temp = create_regular_grid_from_data(df_for_grid, n_points_lat_out, n_points_lon_out)
                grid_df = df_for_grid.iloc[grid_indices].reset_index(drop=True)
                
                # Verify grid has valid coordinates
                coord_cols = ['x_coord', 'y_coord', 'z_coord']
                grid_df = grid_df.dropna(subset=coord_cols).reset_index(drop=True)
                
                if len(grid_df) < 2:
                    print(f"Warning: Not enough valid grid points after filtering, skipping...")
                    continue
                
                output_grid_created = True
                print(f"Output grid created with {len(grid_df)} valid points\n")
            
            # Cache output graph across samples for same year
            cached_output = None
            
            for sample_idx in range(samples_per_year):
                print(f"\r[{year_idx + 1}/{len(years)}] {year}: {sample_idx + 1}/{samples_per_year}", end='')
                
                if cache_region_data:
                    df = cached_df.copy()
                else:
                    min_lon_g, min_lat_g, max_lon_g, max_lat_g = sampling_bounds
                    
                    if bbox_size_deg is not None:
                        # Random sub-region sampling
                        margin = 0.01
                        min_lon_g += margin
                        min_lat_g += margin
                        max_lon_g -= margin
                        max_lat_g -= margin
                        
                        if max_lon_g - min_lon_g < bbox_size_deg or max_lat_g - min_lat_g < bbox_size_deg:
                            print(f"\nWarning: bbox_size_deg ({bbox_size_deg}) too large for region, skipping...")
                            continue
                        
                        min_lon = random.uniform(min_lon_g, max_lon_g - bbox_size_deg)
                        min_lat = random.uniform(min_lat_g, max_lat_g - bbox_size_deg)
                        bbox = (min_lon, min_lat, min_lon + bbox_size_deg, min_lat + bbox_size_deg)
                    else:
                        bbox = (min_lon_g, min_lat_g, max_lon_g, max_lat_g)
                    
                    df = prepare_dataframe(tif_path, bbox)
                
                # Create sample
                data_sample, cached_output = create_subgraph_sample(
                    df_full=df,
                    year=year,
                    n_points_input_min=n_points_input_min,
                    n_points_input_max=n_points_input_max,
                    grid_df=grid_df,
                    k_in=k_in,
                    k_out=k_out,
                    k_cross=k_cross,
                    perplexity=perplexity,
                    device=device,
                    filename=os.path.basename(tif_path),
                    cached_output=cached_output
                )
                
                writer.append(data_sample)
            
            print(f"\r[{year_idx + 1}/{len(years)}] {year}: {samples_per_year}/{samples_per_year} ✓")
    
    finally:
        writer.close()
    
    print(f"\n✓ Subgraph dataset creation complete: {output_path}")


def validate_subgraph_config(config: Dict[str, Any]) -> None:
    """
    Validate subgraph configuration.
    """
    required_fields = [
        'k_in', 'k_out', 'k_cross', 'perplexity',
        'n_points_input_min', 'n_points_input_max', 'n_points_lat_out', 'n_points_lon_out'
    ]
    
    # Check if using manual split or automatic split
    has_manual_split = 'train_years' in config or 'test_years' in config
    has_years_list = 'years' in config
    
    if not has_manual_split and not has_years_list:
        raise KeyError("Must provide either 'years' (for automatic split) or 'train_years'/'test_years' (for manual split)")
    
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Required configuration field missing: {field}")
    
    # Validate automatic split case
    if has_years_list and not has_manual_split:
        if not config['years']:
            raise ValueError("years list cannot be empty")
        if 'samples_per_year' not in config:
            raise KeyError("samples_per_year required when using automatic split")
        if config['samples_per_year'] <= 0:
            raise ValueError(f"samples_per_year must be positive")
    
    # Validate manual split case
    if has_manual_split:
        if 'train_years' not in config or 'test_years' not in config:
            raise KeyError("Both train_years and test_years required for manual split")
        if 'train_samples_per_year' not in config or 'test_samples_per_year' not in config:
            raise KeyError("Both train_samples_per_year and test_samples_per_year required for manual split")
        if config['train_samples_per_year'] <= 0:
            raise ValueError(f"train_samples_per_year must be positive")
        if config['test_samples_per_year'] <= 0:
            raise ValueError(f"test_samples_per_year must be positive")
    
    # Validate k values
    if config['k_in'] <= 0:
        raise ValueError(f"k_in must be positive")
    if config['k_out'] <= 0:
        raise ValueError(f"k_out must be positive")
    if config['k_cross'] <= 0:
        raise ValueError(f"k_cross must be positive")
    if config['perplexity'] <= 0:
        raise ValueError(f"perplexity must be positive")
    if config['n_points_input_min'] <= 0:
        raise ValueError(f"n_points_input_min must be positive")
    if config['n_points_input_max'] < config['n_points_input_min']:
        raise ValueError(f"n_points_input_max must be >= n_points_input_min")
    if config['n_points_lat_out'] <= 0:
        raise ValueError(f"n_points_lat_out must be positive")
    if config['n_points_lon_out'] <= 0:
        raise ValueError(f"n_points_lon_out must be positive")
