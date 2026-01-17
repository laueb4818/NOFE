#!/usr/bin/env python3
"""
Test script for interpolated workflow with ERA5 data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'ERA5' / 'scripts'))

import torch
from utilities_era5 import prepare_dataframe, create_interpolated_sample

def test_interpolated_sample():
    """Test creating an interpolated sample."""
    print("Testing interpolated sample creation...")
    
    # Setup
    device = torch.device("cpu")
    date = "2018-01-01"
    time = "00:00"
    vars_list = ['w', 'v']  # Use fewer variables for faster testing
    
    # Load data
    print(f"Loading data for {date}...")
    try:
        df = prepare_dataframe(
            date=date,
            time=time,
            vars=vars_list,
            datafolder_path="../data"  # Adjust path as needed
        )
        print(f"Loaded {len(df)} data points")
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found. Adjust path or use available data.")
        print(f"  {e}")
        return False
    
    # Create interpolated sample
    print("\nCreating interpolated sample...")
    try:
        sample = create_interpolated_sample(
            df_full=df,
            date=date,
            time=time,
            lat_min=50.0,
            lat_max=60.0,
            lon_min=5.0,
            lon_max=15.0,
            n_lat_out=10,
            n_lon_out=10,
            n_points_input_min=100,
            n_points_input_max=200,
            k=20,
            perplexity=10,
            device=device,
            filename=f"{date}.nc"
        )
        
        print("\n✓ Sample created successfully!")
        print(f"  Total nodes: {sample.x.shape[0]}")
        print(f"  Input nodes: {sample.n_input}")
        print(f"  Output nodes: {sample.n_output}")
        print(f"  Node features shape: {sample.node_features.shape}")
        print(f"  Edges: {sample.edge_index.shape[1]}")
        print(f"  GPS coords shape: {sample.gps.shape}")
        print(f"  Filename: {sample.filename}")
        
        # Verify shapes
        assert sample.x.shape[0] == sample.n_input + sample.n_output
        assert sample.node_features.shape[0] == sample.x.shape[0]
        assert sample.gps.shape == (sample.x.shape[0], 2)
        
        print("\n✓ All shape checks passed!")
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
