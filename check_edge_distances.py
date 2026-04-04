#!/usr/bin/env python3
"""
Check edge distances to see if there are zeros that would cause division by zero issues.
"""

import sys
import torch
from pathlib import Path

sys.path.append("src/")
from dataset_utilities import GraphLMDBReader

def check_edge_distances(sample, sample_idx):
    """Check edge distances for potential division by zero issues."""
    
    if not hasattr(sample, 'edge_dist') or sample.edge_dist is None:
        print(f"Sample {sample_idx}: No edge_dist")
        return
    
    if not hasattr(sample, 'edge_index') or sample.edge_index is None:
        print(f"Sample {sample_idx}: No edge_index")
        return
    
    edge_dist = sample.edge_dist.squeeze()
    edge_index = sample.edge_index
    
    num_nodes = sample.x.shape[0] if hasattr(sample, 'x') else edge_index.max().item() + 1
    
    # Compute max_dist_per_node as the training code would
    max_dist_per_node = torch.zeros(num_nodes, device=edge_dist.device)
    max_dist_per_node.scatter_reduce_(
        0, edge_index[0], edge_dist, reduce="amax"
    )
    
    # Check for zeros
    zero_mask = (max_dist_per_node == 0)
    num_zeros = zero_mask.sum().item()
    
    # Check for very small values
    small_mask = (max_dist_per_node > 0) & (max_dist_per_node < 1e-10)
    num_small = small_mask.sum().item()
    
    # Stats on edge_dist
    min_edge_dist = edge_dist.min().item()
    max_edge_dist = edge_dist.max().item()
    num_zero_edges = (edge_dist == 0).sum().item()
    
    print(f"\nSample {sample_idx}:")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Num edges: {len(edge_dist)}")
    print(f"  Edge distances:")
    print(f"    Min: {min_edge_dist:.10f}")
    print(f"    Max: {max_edge_dist:.10f}")
    print(f"    Num zero edges: {num_zero_edges} / {len(edge_dist)}")
    print(f"  Max distance per node:")
    print(f"    Nodes with max_dist = 0: {num_zeros} / {num_nodes}")
    print(f"    Nodes with max_dist < 1e-10: {num_small} / {num_nodes}")
    
    if num_zeros > 0:
        print(f"    ❌ WARNING: {num_zeros} nodes have max_dist_per_node = 0!")
        print(f"       This WILL cause division by zero and NaN!")
        # Show which nodes
        zero_nodes = torch.where(zero_mask)[0]
        print(f"       Zero nodes: {zero_nodes[:10].tolist()}{'...' if len(zero_nodes) > 10 else ''}")
    
    if num_small > 0:
        print(f"    ⚠️  {num_small} nodes have very small max_dist_per_node < 1e-10")
    
    if num_zeros == 0 and num_small == 0:
        print(f"    ✓ All nodes have reasonable max_dist_per_node")
    
    return num_zeros > 0

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
        print(f"Dataset: {dataset_path}")
        print(f"{'='*70}")
        
        reader = GraphLMDBReader(str(path))
        print(f"Total samples: {len(reader)}")
        
        # Check first 10 samples
        num_to_check = min(10, len(reader))
        found_issues = False
        
        for i in range(num_to_check):
            has_zeros = check_edge_distances(reader[i], i)
            if has_zeros:
                found_issues = True
        
        if found_issues:
            print(f"\n❌ FOUND DIVISION BY ZERO ISSUES in {dataset_path}")
        else:
            print(f"\n✓ No division by zero issues found in first {num_to_check} samples")

if __name__ == "__main__":
    main()
