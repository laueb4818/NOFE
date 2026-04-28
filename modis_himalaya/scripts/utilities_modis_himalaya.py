#%%
import sys 
import os
import numpy as np
import pandas as pd
import json
import torch
import rasterio
from rasterio.windows import Window
from torch_geometric.data import Data
from torch_scatter import scatter
from pyproj import Transformer


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))
#os.chdir(os.path.dirname(__file__))

from dataset_utilities import (
    GraphLMDBWriter,
    GraphLMDBReader
)
from generator_utility import (
    Graph_Generator,
    track_time,
    _function_timings,
    device,
    clear_memory_cache
)


REGION_EXTENT_KM = 2000 * 1000

REGION_X_MIN = -1_500_000
REGION_X_MAX = 1_500_000
REGION_X_CENTER = (REGION_X_MIN + REGION_X_MAX) / 2
REGION_X_HALF_RANGE = (REGION_X_MAX - REGION_X_MIN) / 2

REGION_Y_MIN = 2_500_000
REGION_Y_MAX = 4_500_000
REGION_Y_CENTER = (REGION_Y_MIN + REGION_Y_MAX) / 2
REGION_Y_HALF_RANGE = (REGION_Y_MAX - REGION_Y_MIN) / 2


#%%
class MODIS_Himalaya_Graph_Generator(Graph_Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    


    def get_available_timestamps(self, min_date=None, max_date=None, min_file_size_mb=None):
        if min_file_size_mb is None:
            min_file_size_mb = 250
        min_bytes = min_file_size_mb * 1024 * 1024

        def parse_config_date(date_str):
            if date_str is None:
                return None
            return date_str.replace('-', '')

        min_date_parsed = parse_config_date(min_date)
        max_date_parsed = parse_config_date(max_date)

        timestamps = set()
        for f in os.listdir(self.directory):
            if not f.startswith('brighttemp') or not f.endswith('.tif'):
                continue
            bright_path = os.path.join(self.directory, f)
            if os.path.getsize(bright_path) < min_bytes:
                continue
            ts = f.replace('brighttemp_1KM_grid_modis_', '').replace('.tif', '')

            ts_date = ts[:8]

            if min_date_parsed and ts_date < min_date_parsed:
                continue
            if max_date_parsed and ts_date > max_date_parsed:
                continue

            refl_file = f'reflectance_1KM_grid_modis_{ts}.tif'
            refl_path = os.path.join(self.directory, refl_file)
            if not os.path.exists(refl_path):
                continue

            timestamps.add(ts)

        sorted_timestamps = sorted(timestamps)
        print(f"Found {len(sorted_timestamps)} valid timestamps using date range {min_date} to {max_date}, filesize >= {min_file_size_mb} MB")
        return sorted_timestamps


    @track_time
    def load_data(self, timestamp:str=None, bbox:tuple=None):
        bright_file = f'brighttemp_1KM_grid_modis_{timestamp}.tif'
        refl_file = f'reflectance_1KM_grid_modis_{timestamp}.tif'
        
        bright_path = os.path.join(self.directory, bright_file)
        refl_path = os.path.join(self.directory, refl_file)
        
        if not os.path.exists(bright_path):
            raise FileNotFoundError(f"Brightness file not found: {bright_path}")
        if not os.path.exists(refl_path):
            raise FileNotFoundError(f"Reflectance file not found: {refl_path}")
        
        self.meta = Data(
            source_file_bright=os.path.abspath(bright_path),
            source_file_refl=os.path.abspath(refl_path),
            timestamp=timestamp
        )
        
        with rasterio.open(bright_path) as src_bright:
            bright = src_bright.read()
        
        with rasterio.open(refl_path) as src_refl:
            refl = src_refl.read()
        
        combined = np.concatenate([bright, refl], axis=0)
        
        for c in range(combined.shape[0]):
            channel = combined[c]
            mean = np.nanmean(channel)
            std = np.nanstd(channel)
            if std == 0:
                std = 1
            combined[c] = (channel - mean) / (std + 1e-8)
        
        rows, cols = combined.shape[1], combined.shape[2]
        
        if bbox is not None:
            row_min, col_min, row_max, col_max = bbox
            row_min = max(0, row_min)
            col_min = max(0, col_min)
            row_max = min(rows, row_max)
            col_max = min(cols, col_max)
            self.da = combined[:, row_min:row_max, col_min:col_max]
            self.crop_offset = (row_min, col_min)
        else:
            self.da = combined
            self.crop_offset = (0, 0)
        
        rows, cols = self.da.shape[1], self.da.shape[2]
        self.grid_row_indices = np.arange(rows)
        self.grid_col_indices = np.arange(cols)
        
        transform = src_bright.transform
        self.transform = transform
        self.crs = src_bright.crs
        
        to_wgs84 = Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)
        
        rows_grid, cols_grid = np.meshgrid(
            np.arange(rows) + 0.5,
            np.arange(cols) + 0.5,
            indexing='ij'
        )
        
        xs_utm, ys_utm = transform * (cols_grid, rows_grid)
        
        lons, lats = to_wgs84.transform(xs_utm, ys_utm)
        
        self.grid_longitudes = lons[0, :]
        self.grid_latitudes = lats[:, 0]


    def norm_coords(self, indices):
        rows = indices[:, 0]
        cols = indices[:, 1]
        
        global_rows = rows + self.crop_offset[0]
        global_cols = cols + self.crop_offset[1]
        
        xs = np.zeros(len(indices))
        ys = np.zeros(len(indices))
        
        for i, (r, c) in enumerate(zip(global_rows, global_cols)):
            x, y = self.transform * (c + 0.5, r + 0.5)
            xs[i] = x
            ys[i] = y
        
        normed_coords = np.zeros((len(indices), 3))
        normed_coords[:, 0] = (xs - REGION_X_CENTER) / REGION_X_HALF_RANGE
        normed_coords[:, 1] = (ys - REGION_Y_CENTER) / REGION_Y_HALF_RANGE
        normed_coords[:, 2] = np.zeros(len(indices))
        
        return normed_coords


    @track_time
    def generate_normed_sample(self, N:int=None, method:str=None):
        n_channels, n_row, n_col = self.da.shape
        
        total_pixels = n_row * n_col
        
        if N is None:
            N = self.N
        if method is None:
            method = self.generator_method
        
        if N >= total_pixels:
            print(f"requested number of points larger than dataset -> returning full dataset of {total_pixels} points.")
            N = total_pixels
        
        if method == "random":
            flat_indices = np.random.choice(total_pixels, size=N, replace=False)
            indices = np.column_stack(np.unravel_index(flat_indices, (n_row, n_col)))
            indices = indices[np.lexsort((indices[:, 1], indices[:, 0]))]
        
        elif method == "interpolate":
            pass
        
        elif method == "regular_subset":
            aspect = n_row / n_col
            n_row_new = int(round(np.sqrt(N * aspect)))
            n_col_new = int(round(N / n_row_new))
            
            row_inds = np.arange(n_row)
            row_inds_subset = row_inds[np.linspace(0, len(row_inds) - 1, n_row_new, dtype=int)]
            
            col_inds = np.arange(n_col)
            col_inds_subset = col_inds[np.linspace(0, len(col_inds) - 1, n_col_new, dtype=int)]
            
            indices1 = np.repeat(row_inds_subset, n_col_new)
            indices2 = np.tile(col_inds_subset, n_row_new)
            
            indices = np.vstack([indices1, indices2]).T
        
        else:
            print(f'"{method}" is an invalid sample method!')
            return None, None, None
        
        normed_values = torch.as_tensor(self.da[:, indices[:, 0], indices[:, 1]].T.copy())
        
        latitudes = self.grid_latitudes[indices[:, 0]]
        longitudes = self.grid_longitudes[indices[:, 1]]
        
        gps = np.column_stack((latitudes, longitudes))
        self.gps = gps
        
        meta = Data(gps=gps, timestamp=self.meta['timestamp'])
        
        normed_coords = self.norm_coords(indices)
        
        return normed_values, normed_coords, meta


    def synthetic_graph(self, N:int, k:int, gen_feature_mode:str="channel_only"):
        n_channels, n_row, n_col = self.da.shape
        
        aspect = n_row / n_col
        n_row_new = int(round(np.sqrt(N * aspect)))
        n_col_new = int(round(N / n_row_new))
        
        row_inds = np.linspace(0, n_row - 1, n_row_new, dtype=int)
        col_inds = np.linspace(0, n_col - 1, n_col_new, dtype=int)
        
        indices1 = np.repeat(row_inds, n_col_new)
        indices2 = np.tile(col_inds, n_row_new)
        indices = np.vstack([indices1, indices2]).T
        
        latitudes = self.grid_latitudes[indices[:, 0]]
        longitudes = self.grid_longitudes[indices[:, 1]]
        
        normed_coords = self.norm_coords(indices)
        
        normed_coords = torch.as_tensor(normed_coords, device=device, dtype=torch.float32)
        dummy_features = torch.zeros(len(latitudes), n_channels, device=device, dtype=torch.float32)
        
        meta = Data(gps=np.column_stack((latitudes, longitudes)), synthetic=True)
        
        edge_indices, edge_dist, _ = self.get_edges(
            source=normed_coords.cpu().numpy(), 
            target=normed_coords.cpu().numpy(), 
            k=k, 
            perplexity=0, 
            mode="self_search"
        )
        
        edge_indices = torch.as_tensor(edge_indices, device=device, dtype=torch.long)
        edge_dist = torch.as_tensor(edge_dist, device=device, dtype=torch.float32)
        
        node_features = self.get_node_features(
            normed_coords=normed_coords, 
            normed_values=dummy_features, 
            mode=gen_feature_mode
        )
        
        edge_attributes = self.get_edge_attributes(
            source_coords=normed_coords,
            target_coords=normed_coords,
            edge_indices=edge_indices,
            edge_dist=edge_dist,
            k=k,
            include_feat_diff=False
        )
        
        return Data(
            x=node_features, 
            edge_index=edge_indices, 
            edge_attr=edge_attributes, 
            edge_dist=edge_dist, 
            input_affinities=None, 
            sigmas=None, 
            normed_coords=normed_coords, 
            normed_values=dummy_features, 
            meta=meta
        )