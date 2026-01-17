#!/usr/bin/env python3
"""
Test script for interpolated workflow with alphaEarth data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'alphaEarth' / 'scripts'))

import torch
from utilities_alphaearth import prepare_dataframe, create_interpolated_sample

def test_interpolated_sample():
    """Test creating an interpolated sample."""
    print("Testing interpolated sample creation...")
    
    # Setup
    device = torch.device("cpu")
    year = 2019
    data_path = "/Volumes/T7-storage/alphaEarth/data"
    tif_path = f"{data_path}/sat_emb_{year}.tif"
    
    # Define a small test region (Norway)
    lat_min, lat_max = 59.0, 59.5
    lon_min, lon_max = 10.0, 10.5
    bbox = (lon_min, lat_min, lon_max, lat_max)
    
    # Load data
    print(f"Loading data for {year} from region...")
    print(f"  Bbox: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    try:
        df = prepare_dataframe(tif_path, bbox)
        print(f"Loaded {len(df)} data points")
    except Exception as e:
        print(f"ERROR: Could not load data")
        print(f"  {type(e).__name__}: {e}")
        return False
    
    # Create interpolated sample
    print("\nCreating interpolated sample...")
    try:
        sample = create_interpolated_sample(
            df_full=df,
            year=year,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            n_lat_out=10,
            n_lon_out=10,
            n_points_input_min=100,
            n_points_input_max=200,
            k=20,
            perplexity=10,
            device=device,
            filename=f"sat_emb_{year}.tif"
        )
        
        print("\n✓ Sample created successfully!")
        print(f"  Total nodes: {sample.x.shape[0]}")
        print(f"  Input nodes: {sample.n_input}")
        print(f"  Output nodes: {sample.n_output}")
        print(f"  Node position shape: {sample.x.shape}")
        print(f"  Node features shape: {sample.node_features.shape}")
        print(f"  Edges: {sample.edge_index.shape[1]}")
        print(f"  GPS coords shape: {sample.gps.shape}")
        print(f"  Year: {sample.year}")
        print(f"  Filename: {sample.filename}")
        
        # Verify shapes
        assert sample.x.shape[0] == sample.n_input + sample.n_output, \
            f"Total nodes mismatch: {sample.x.shape[0]} != {sample.n_input} + {sample.n_output}"
        assert sample.node_features.shape[0] == sample.x.shape[0], \
            f"Features shape mismatch: {sample.node_features.shape[0]} != {sample.x.shape[0]}"
        assert sample.gps.shape == (sample.x.shape[0], 2), \
            f"GPS shape mismatch: {sample.gps.shape} != {(sample.x.shape[0], 2)}"
        assert sample.x.shape[1] == 3, \
            f"Coordinate dimension should be 3: {sample.x.shape[1]}"
        
        # Verify coordinates are normalized (on unit sphere)
        norms = torch.norm(sample.x, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Coordinates not on unit sphere: min={norms.min()}, max={norms.max()}"
        
        # Verify output grid is in the specified bounds
        output_gps = sample.gps[sample.n_input:].numpy()
        assert output_gps[:, 0].min() >= lat_min - 1e-6, "Output lat below min"
        assert output_gps[:, 0].max() <= lat_max + 1e-6, "Output lat above max"
        assert output_gps[:, 1].min() >= lon_min - 1e-6, "Output lon below min"
        assert output_gps[:, 1].max() <= lon_max + 1e-6, "Output lon above max"
        
        print("\n✓ All checks passed!")
        print(f"  Coordinate norms: min={norms.min():.6f}, max={norms.max():.6f}")
        print(f"  Output grid bounds: lat [{output_gps[:, 0].min():.3f}, {output_gps[:, 0].max():.3f}], " +
              f"lon [{output_gps[:, 1].min():.3f}, {output_gps[:, 1].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR during sample creation:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_interpolated_sample()
    sys.exit(0 if success else 1)
