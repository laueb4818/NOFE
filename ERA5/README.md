# ERA5 Dataset Creation

This directory contains scripts for creating graph-based datasets from ERA5 weather data for neural operator training.

## Structure

```
ERA5/
├── scripts/                   # Dataset creation scripts
│   ├── create_dataset.py      # Main script for creating datasets
│   └── utilities_era5.py      # Utility functions (self-contained)
├── dataset_configs/           # Configuration files for datasets
│   └── example_config.json    # Example configuration
├── datasets/                  # Output directory for LMDB datasets
├── model_configs/             # Model configuration files
└── training_configs/          # Training configuration files
```

## Quick Start

### 1. Create a Dataset Configuration

Create a JSON file in `dataset_configs/` with the following structure:

```json
{
    "n_points_min": 500,
    "n_points_max": 2000,
    "samples_per_date": 20,
    "k": 30,
    "perplexity": 10,
    "lifting_bool": false,
    "vars": ["w", "v", "z", "q", "r", "t", "cc", "d", "o3", "pv", "u", "vo"],
    "dates": ["2018-01-01", "2018-01-02", "2018-01-03"]
}
```

**Configuration Parameters:**
- `n_points_min/max`: Range of points to sample per region
- `samples_per_date`: Number of samples to create per date
- `k`: Number of nearest neighbors for graph construction
- `perplexity`: Perplexity parameter for adaptive affinity computation
- `lifting_bool`: Whether to apply feature lifting (outer products)
- `vars`: List of ERA5 variable short names to include
- `dates`: List of dates in YYYY-MM-DD format

### 2. Run Dataset Creation

From the `scripts/` directory:
```bash
cd scripts/
python create_dataset.py <config_name>
```

Or from the parent directory:
```bash
python scripts/create_dataset.py <config_name>
```

**Options:**
- `<config_name>`: Name of config file (without .json extension)
- `--no-test`: Skip test dataset creation (only create training set)

**Examples:**
```bash
# Create both train and test datasets
python create_dataset.py example_config

# Create only training dataset
python create_dataset.py example_config --no-test
```

### 3. Output

The script creates LMDB datasets in the `datasets/` directory:
- `<config_name>_train.lmdb`: Training dataset (80% of dates)
- `<config_name>_test.lmdb`: Test dataset (20% of dates)

## Dataset Format

Each sample in the LMDB dataset is a PyTorch Geometric `Data` object with:

- **`x`**: Node features `[N, D]` - normalized atmospheric variables (optionally lifted)
- **`edge_index`**: Graph connectivity `[2, N*k]` - k-NN graph structure
- **`edge_attr`**: Edge attributes `[N*k, E]` - spatial distances, directions, and temporal encoding
- **`locations`**: Original coordinates `[N, 2]` - lat/lon positions
- **`input_affinities`**: Adaptive affinities `[N, k]` - Gaussian affinities based on feature similarity
- **`date`**: Date string for reference

## Loading Datasets

```python
from scripts.utilities_era5 import GraphLMDBReader

# Load dataset
reader = GraphLMDBReader('datasets/example_config_train.lmdb')
print(f'Dataset size: {len(reader)} samples')

# Load a sample
sample = reader[0]
print(f'Node features: {sample.x.shape}')
print(f'Edge index: {sample.edge_index.shape}')
```

## Features

### Modular Design
- **Self-contained**: `utilities_era5.py` has no external dependencies on other project files
- **Clean separation**: Core utilities separate from high-level dataset creation logic
- **Well-documented**: Comprehensive docstrings for all functions

### Dataset Creation Pipeline
1. Load ERA5 data from NetCDF files
2. Apply Cartesian projection to coordinates
3. Random spatial subsetting (lat/lon bounding boxes)
4. Random point sampling within regions
5. k-NN graph construction in Cartesian space
6. Feature normalization with optional lifting
7. Edge attribute computation (spatial + temporal encoding)
8. Adaptive affinity computation using perplexity
9. Storage in LMDB format for efficient loading

### Key Features
- **Train/test split**: Automatic 80/20 split by dates
- **Random sampling**: Each sample has random spatial extent and point count
- **Adaptive affinities**: Per-point sigma values matched to target perplexity
- **Temporal encoding**: Sine/cosine encoding of day of year
- **Device support**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- xarray
- pandas
- numpy
- faiss-cpu or faiss-gpu
- lmdb
- pyproj

## Data Location

ERA5 NetCDF files should be located at:
```
/Volumes/T7-storage/ERA5/data/
```

Files should be named: `YYYY-MM-DD.nc`
