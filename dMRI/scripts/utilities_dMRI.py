#%%
import sys 
import os
import numpy as np
import torch
from torch_geometric.data import Data
import nibabel as nib
from dipy.core.gradients import gradient_table
import random




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
    clear_memory_cache
)

class dMRI_Graph_Generator(Graph_Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
    
    def load_segmentation(self, path:str):
        """Load aparc+aseg parcellation and store brain-masked label array."""
        seg_img = nib.load(path + "aparc+aseg.nii.gz")
        seg = seg_img.get_fdata().astype(np.int32)
        mask = self.meta.mask.numpy()          # full flat mask (X*Y*Z,)
        self.seg_labels = seg.reshape(-1)[mask] # brain-only labels (N_brain,)

    def load_data(self, path:str, brain_only=True):
        
        dwi_img = nib.load(path + "data.nii.gz")
        data = dwi_img.get_fdata().astype(np.float32)  # shape: (X,Y,Z,N_directions)
        
        # get measurement parameters
        bvals = np.loadtxt(path + "bvals")
        bvecs = np.loadtxt(path + "bvecs")
        gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=100)
        
        # Load brain mask from file
        mask_img = nib.load(path + "nodif_brain_mask.nii.gz")
        mask = mask_img.get_fdata().astype(bool)  # shape: (X,Y,Z)
        
        # Voxel coordinates
        X, Y, Z = mask.shape
        coords = np.stack(np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij'), axis=-1)
        
        if brain_only:
            # Flatten and select only brain voxels
            coords_flat = coords.reshape(-1, 3)[mask.reshape(-1)]
            vectors_flat = data.reshape(-1, data.shape[-1])[mask.reshape(-1)]
        else:
            # Flatten only
            coords_flat = coords.reshape(-1, 3)
            vectors_flat = data.reshape(-1, data.shape[-1])
        
        # Convert to torch tensors
        pos = torch.tensor(coords_flat, dtype=torch.float)
        x = torch.tensor(vectors_flat, dtype=torch.float)
        
        # normlization bias: mean value for b0 field
        b0_bias = x[:,gtab.b0s_mask].mean(dim=1, keepdim=True) + 10e-8
        x_norm = x / b0_bias
        
        # normalize positions to [0,1]
        pos = pos - pos.min(dim=0).values
        pos = pos / pos.max(dim=0).values
        
        mask_tensor = torch.tensor(mask.reshape(-1), dtype=torch.bool)
        
        meta = Data(mask = mask_tensor, file_path=path)
        
        self.da = x_norm
        self.normed_coords = pos
        self.meta = meta
    
    
    @track_time
    def generate_normed_sample(self, N:int=None, method:str=None):
        """
        This method generates a sample from initially loaded data self.da
        
        :param self: Description
        :param N: number of points to sample
        :type N: int
        :param method: 
        'random': randomly samples N points
        'interpolate': not defined yet
        'regular_subset': based on number of points N, the method selects a subset of the data that matches a regular grid structure
        :type method: str
        """
        
        #n_channels, n_lat, n_lon = self.da.shape
        N_total = self.da.shape[0]
        '''
        if N==None:
            N=self.N
        if method==None:
            method=self.generator_method
        '''
        
        if N >= N_total:
            print(f"requested number of points larger then dataset\n-> return full dataset of {N_total} points.")
            N = N_total
        
        if method=="random":
            indices = random.sample(range(N_total), N)
            # Convert to tensor to avoid CUDA indexing errors
            indices = torch.tensor(indices, dtype=torch.long)

        elif method=="region_stratified":
            # Equal sampling from each coordinate-based brain lobe.
            # Ensures all regions are represented regardless of their volume.
            y = self.normed_coords[:, 1].numpy()
            z = self.normed_coords[:, 2].numpy()
            lobe_masks = {
                'frontal':     y > 0.60,
                'parietal':    (y > 0.35) & (y < 0.85) & (z > 0.62),
                'temporal':    (y > 0.20) & (y < 0.70) & (z < 0.55),
                'occipital':   y < 0.25,
                'subcortical': (y > 0.35) & (y < 0.65) & (z > 0.28) & (z < 0.58),
            }
            n_regions  = len(lobe_masks)
            per_region = N // n_regions
            remainder  = N % n_regions
            all_indices = []
            for i, (_, mask) in enumerate(lobe_masks.items()):
                pool     = np.where(mask)[0]
                n_sample = per_region + (1 if i < remainder else 0)
                if len(pool) == 0:
                    continue
                chosen = pool if len(pool) <= n_sample else \
                         np.random.choice(pool, size=n_sample, replace=False)
                all_indices.append(chosen)
            indices = torch.tensor(np.concatenate(all_indices), dtype=torch.long)

        elif method=="interpolate":
            pass

        elif method=="regular_subset":
            pass

        else:
            print(f'"{method}" is an invalid sample method!')
        
        #self.normed_values = torch.tensor(self.da[:, indices[:,0], indices[:,1]].T)
        # more efficient. previously: normed_values = torch.tensor(self.da[:, indices[:,0], indices[:,1]].T)
        normed_values = self.da[indices, :].detach().clone()
        normed_coords = self.normed_coords[indices, :].detach().clone()
        meta = self.meta.detach().clone()
        self.last_indices = indices  # expose for region-based embedding extraction

        return normed_values, normed_coords, meta

