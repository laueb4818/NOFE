#!/usr/bin/env python3
"""
Dataset Creation Script for AlphaEarth Satellite Data
======================================================

Creates train and test LMDB datasets with graph-structured samples from AlphaEarth
GeoTIFF data. Automatically includes all bands (no variable selection needed).

Key differences from ERA5:
- Reads GeoTIFF files instead of NetCDF
- Uses years instead of dates
- Automatically includes all bands

Usage:
    python create_dataset.py <config_name> [--no-test] [--data_path PATH]
    
Examples:
    python create_dataset.py example_config
    # Uses default: ../data/
    
    python create_dataset.py example_config --data_path /path/to/alphaEarth/data
    # Uses custom path
"""

import sys
import os
import argparse
from pathlib import Path

from alphaEarth.scripts.old_utilities_alphaearth import (
    setup_device,
    load_config,
    validate_config,
    split_train_test,
    create_lmdb_dataset,
    print_dataset_summary,
    set_seed,
)


def main():
    """Main entry point for dataset creation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create AlphaEarth LMDB datasets from configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config_name",
        help="Name of config file in dataset_configs/ (without .json extension)"
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
    
    # Setup paths (relative to parent directory since we're in scripts/)
    parent_dir = Path(__file__).parent.parent
    dataset_config_path = parent_dir / "dataset_configs"
    dataset_output_path = parent_dir / "datasets"
    dataset_output_path.mkdir(exist_ok=True)
    
    # Data path: use provided path or default to ../data/
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
        validate_config(config)
    except (KeyError, ValueError) as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}\n")
    
    # Determine train/test split
    has_manual_split = 'train_years' in config and 'test_years' in config
    
    if has_manual_split:
        # Use manual split from config
        train_years = config['train_years']
        test_years = config['test_years']
        train_samples = config.get('train_samples_per_year')
        test_samples = config.get('test_samples_per_year')
    else:
        # Use automatic split
        all_years = config['years']
        train_years, test_years = split_train_test(
            all_years,
            train_ratio=0.8,
            shuffle=True,
            random_seed=seed
        )
        train_samples = None
        test_samples = None
    
    # Print dataset summary
    print_dataset_summary(config_name, train_years, test_years, config)
    
    # Create training dataset
    train_output = dataset_output_path / f"{config_name}_train.lmdb"
    create_lmdb_dataset(
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
        create_lmdb_dataset(
            config=config,
            years=test_years,
            output_path=str(test_output),
            device=device,
            data_path=str(data_path),
            samples_per_year=test_samples
        )
    
    print("\n" + "="*60)
    print("Dataset creation finished successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
