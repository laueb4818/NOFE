#!/usr/bin/env python3
"""
Check LMDB datasets for NaN, Inf, and numerical issues.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append("src/")
from utilities import GraphLMDBReader

def check_tensor_stats(name, tensor):
    """Check tensor for NaN, Inf, and return statistics."""
    if tensor is None:
        return None
    
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor)
    
    stats = {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'nan_count': torch.isnan(tensor).sum().item(),
        'inf_count': torch.isinf(tensor).sum().item(),
        'min': tensor[torch.isfinite(tensor)].min().item() if torch.isfinite(tensor).any() else float('nan'),
        'max': tensor[torch.isfinite(tensor)].max().item() if torch.isfinite(tensor).any() else float('nan'),
        'mean': tensor[torch.isfinite(tensor)].mean().item() if torch.isfinite(tensor).any() else float('nan'),
    }
    
    return stats

def check_sample(graph, sample_idx, verbose=False):
    """Check a single graph sample."""
    issues = []
    stats = {}
    
    # Check all tensor attributes
    attrs_to_check = ['x', 'edge_index', 'edge_attr', 'input_affinities', 
                      'sigmas', 'edge_dist', 'normed_coords', 'normed_values']
    
    for attr in attrs_to_check:
        if hasattr(graph, attr) and getattr(graph, attr) is not None:
            tensor = getattr(graph, attr)
            s = check_tensor_stats(attr, tensor)
            stats[attr] = s
            
            if s['nan_count'] > 0:
                issues.append(f"{attr}: {s['nan_count']} NaNs")
            if s['inf_count'] > 0:
                issues.append(f"{attr}: {s['inf_count']} Infs")
    
    if verbose or issues:
        print(f"\nSample {sample_idx}:")
        if issues:
            print("  ISSUES:", ", ".join(issues))
        for attr, s in stats.items():
            if s:
                print(f"  {attr}: shape={s['shape']}, nan={s['nan_count']}, inf={s['inf_count']}, "
                      f"range=[{s['min']:.6f}, {s['max']:.6f}]")
    
    return len(issues) == 0, stats

def main():
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
        print(f"Checking: {dataset_path}")
        print(f"{'='*70}")
        
        reader = GraphLMDBReader(str(path))
        print(f"Total samples: {len(reader)}")
        
        # Check first 3 samples in detail
        num_to_check = min(3, len(reader))
        all_clean = True
        
        for i in range(num_to_check):
            clean, stats = check_sample(reader[i], i, verbose=True)
            if not clean:
                all_clean = False
        
        # Quick scan of remaining samples
        print(f"\nQuick scanning remaining samples...")
        problem_samples = []
        for i in range(num_to_check, len(reader)):
            clean, _ = check_sample(reader[i], i, verbose=False)
            if not clean:
                problem_samples.append(i)
            if i % 100 == 0:
                print(f"  Checked {i}/{len(reader)}...", end='\r')
        
        print(f"\n  Checked {len(reader)}/{len(reader)}")
        
        if problem_samples:
            print(f"\n  ❌ Found issues in {len(problem_samples)} samples: {problem_samples[:10]}...")
        else:
            print(f"\n  ✓ All samples clean (no NaN/Inf)")

if __name__ == "__main__":
    main()
