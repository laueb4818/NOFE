#!/usr/bin/env python3
"""
Check tensor dtypes in LMDB datasets to diagnose CUDA issues.
"""

import sys
from pathlib import Path

sys.path.append("src/")
from dataset_utilities import GraphLMDBReader

def check_sample_dtypes(sample, sample_idx):
    """Check dtypes of all tensor attributes in a sample."""
    import torch
    
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}")
    print(f"{'='*70}")
    
    # Check all tensor attributes
    attrs_to_check = ['x', 'edge_index', 'edge_attr', 'input_affinities', 
                      'sigmas', 'edge_dist', 'normed_coords', 'normed_values']
    
    for attr in attrs_to_check:
        if hasattr(sample, attr):
            tensor = getattr(sample, attr)
            if tensor is not None:
                print(f"  {attr:20s}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}")
    
    # Check edge_index specifically for int type
    if hasattr(sample, 'edge_index') and sample.edge_index is not None:
        dtype = sample.edge_index.dtype
        is_long = dtype == torch.int64
        print(f"\n  edge_index dtype check:")
        print(f"    dtype: {dtype}")
        print(f"    is torch.int64 (torch.long): {is_long}")
        if not is_long:
            print(f"    ❌ WARNING: edge_index is NOT torch.long!")
            print(f"       This will cause issues on CUDA!")
        else:
            print(f"    ✓ edge_index is torch.long (correct)")

def main():
    import torch
    
    datasets = [
        'ERA5/datasets/random_crop/base_train.lmdb',
        'ERA5/datasets/random_crop/base_test.lmdb',
        'ERA5/datasets/random_crop/dataset1_train.lmdb',
        'ERA5/datasets/random_crop/dataset1_test.lmdb',
    ]
    
    for dataset_path in datasets:
        path = Path(dataset_path)
        if not path.exists():
            print(f"\nSkipping {dataset_path} (not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_path}")
        print(f"{'='*70}")
        
        reader = GraphLMDBReader(str(path))
        print(f"Total samples: {len(reader)}")
        
        # Check first 3 samples
        num_to_check = min(3, len(reader))
        for i in range(num_to_check):
            check_sample_dtypes(reader[i], i)
        
        print()

if __name__ == "__main__":
    main()
