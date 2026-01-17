"""
Subgraph Dataset Creation Script for AlphaEarth Satellite Data
===================================================================

Creates train and test LMDB datasets with two-subgraph structure from AlphaEarth GeoTIFF data.
This is an alternative to create_dataset.py with a different graph topology.

Graph structure:
- Input graph: Random subset of points with k-NN connectivity
- Output graph: All points with k-NN connectivity  
- Cross edges: Connections from output nodes to nearest input nodes

Usage:
    python create_subgraph_dataset.py <config_name> [--no-test] [--data_path PATH]
    
    config_name: Name of config file in dataset_configs_subgraph/ (without .json extension)
    --no-test: Optional flag to skip test dataset creation
    --data_path: Path to AlphaEarth data directory (default: ../data/)

Examples:
    python create_subgraph_dataset.py subgraph_config
    # Uses default: ../data/
"""

# %%
import sys
import os
import argparse
from pathlib import Path
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/alphaEarth/scripts/" )
else:
    device = torch.device("cpu")
print(device)
sys.path.append("../../src/dataset_utilities.py")


from utilities_alphaearth import (
    setup_device,
    load_config,
    validate_subgraph_config,
    split_train_test,
    create_subgraph_lmdb_dataset,
    set_seed,
)

# %%


def print_subgraph_summary(config_name, train_years, test_years, config):
    """Print summary of subgraph dataset configuration."""
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
    print(f"Subgraph Dataset Configuration: {config_name}")
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
    print(f"  k_in (input):       {config['k_in']}")
    print(f"  k_out (output):     {config['k_out']}")
    print(f"  k_cross (cross):    {config['k_cross']}")
    print(f"  Perplexity:         {config['perplexity']}")
    print(f"  Input nodes:        {config['n_points_input_min']}-{config['n_points_input_max']} (random)")
    
    if 'lat_min' in config and 'lat_max' in config and 'lon_min' in config and 'lon_max' in config:
        print(f"\nRegion bounds:")
        print(f"  Latitude:           [{config['lat_min']}, {config['lat_max']}]")
        print(f"  Longitude:          [{config['lon_min']}, {config['lon_max']}]")
    
    print(f"\nYears: {', '.join(map(str, train_years + test_years))}")
    print("=" * 60 + "\n")

# %% Testing around


# %% my own main function
def main():

# %% AI generated previous main function
"""
def main():
    """Main entry point for subgraph dataset creation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create AlphaEarth subgraph LMDB datasets from configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config_name",
        help="Name of config file in dataset_configs_subgraph/ (without .json extension)"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip test dataset creation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to AlphaEarth data directory (default: ../data/ relative to script)"
    )
    
    args = parser.parse_args()
    config_name = args.config_name
    create_test = not args.no_test
    
    # Setup paths
    parent_dir = Path(__file__).parent.parent
    dataset_config_path = parent_dir / "dataset_configs_subgraph"
    dataset_output_path = parent_dir / "datasets_subgraph"
    dataset_output_path.mkdir(exist_ok=True)
    
    # Data path
    if args.data_path:
        data_path = Path(args.data_path).resolve()
    else:
        data_path = (parent_dir / "data").resolve()
    
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    print(f"Data path: {data_path}")
    
    # Setup device
    device = setup_device()
    print(f"Using device: {device}")
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load and validate configuration
    config_file = dataset_config_path / f"{config_name}.json"
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    config = load_config(str(config_file))
    
    try:
        validate_subgraph_config(config)
    except (KeyError, ValueError) as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)
    
    # Set random seed
    seed = config.get('random_seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}\n")
    
    # Determine train/test split
    has_manual_split = 'train_years' in config and 'test_years' in config
    
    if has_manual_split:
        train_years = config['train_years']
        test_years = config['test_years']
        train_samples = config.get('train_samples_per_year')
        test_samples = config.get('test_samples_per_year')
    else:
        all_years = config['years']
        train_years, test_years = split_train_test(
            all_years,
            train_ratio=0.8,
            shuffle=True,
            random_seed=seed
        )
        train_samples = None
        test_samples = None
    
    # Print summary
    print_subgraph_summary(config_name, train_years, test_years, config)
    
    # Create training dataset
    train_output = dataset_output_path / f"{config_name}_train.lmdb"
    create_subgraph_lmdb_dataset(
        config=config,
        years=train_years,
        output_path=str(train_output),
        device=device,
        data_path=str(data_path),
        samples_per_year=train_samples
    )
    
    # Create test dataset
    if create_test and len(test_years) > 0:
        print()
        test_output = dataset_output_path / f"{config_name}_test.lmdb"
        create_subgraph_lmdb_dataset(
            config=config,
            years=test_years,
            output_path=str(test_output),
            device=device,
            data_path=str(data_path),
            samples_per_year=test_samples
        )
    
    print("\n" + "="*60)
    print("Subgraph dataset creation finished successfully!")
    print("="*60)
"""

# %% run main

if __name__ == "__main__":
    main()
