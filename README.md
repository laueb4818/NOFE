# NOFE: Neural Operator Function Embedding

This repository is the official implementation of **NOFE: Neural Operator Function Embedding** (paper link TBD).

> NOFE learns function-to-function mappings via a Graph Kernel Operator, enabling mesh-free evaluation at arbitrary query locations independent of input discretization.
## Requirements

Install dependencies using conda (recommended):

```bash
conda env create -f conda_environment_nofe.yml
conda activate nofe
```

Or with pip:

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.10+
- PyTorch ≥ 2.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) 2.6.1
- `torch-scatter`
- `xarray`, `netcdf4` (ERA5 data)
- `nibabel`, `dipy` (dMRI data)
- `rasterio` (satellite imagery)
- `faiss-cpu` (fast k-NN graph construction)
- `wandb` (optional, experiment tracking)

---

## Dataset Preparation

### ERA5 (Weather)

Download ERA5 data from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) and place it in `ERA5/data/`. Then generate a graph dataset:

```bash
# Fixed spatial region
python ERA5/scripts/create_dataset.py <config_name>

# Random spatial crops (recommended)
python ERA5/scripts/create_dataset_random_crop.py <config_name>

# Subgraph mode (separate input/output graphs)
python ERA5/scripts/create_dataset_random_crop_subgraph.py <config_name>
```

Configuration files are in `ERA5/configurations/`. Example dataset config (`ERA5/configurations/random_crop_subgraph/dataset_configs/baseline.json`):

```json
{
    "k_in": 5, "k_out": 5, "k_cross": 5,
    "N_in_min": 2000, "N_in_max": 8000,
    "N_out_min": 2000, "N_out_max": 10000,
    "input_sampling_method": "random",
    "train_date_start_end": ["2018-01-01", "2018-12-31"],
    "test_date_start_end": ["2019-01-01", "2019-01-31"],
    "vars": ["w", "v", "z", "q", "r", "t", "d", "o3", "pv", "u", "vo"]
}
```

### dMRI

Place dMRI files in `dMRI/data/` and run:

```bash
python dMRI/scripts/create_dataset.py <config_name>
```

---

## Training

Training supports three modes: `standard` (fixed region), `random_crop`, and `subgraph` (disjoint input/output graphs).

### Standard / Random Crop

```bash
python src/train_standard.py \
    --config ERA5/training_configs/<config_name> \
    [--wandb] \
    [--gradient-clipping] \
    [--checkpoint-every 5]
```

### Subgraph Mode

```bash
python src/train_subgraph.py \
    --config ERA5/configurations/random_crop_subgraph/training_configs/base \
    [--wandb]
```

### No-Lifting Variant

```bash
python src/train_standard_nolifting.py \
    --config ERA5/training_configs/<config_name>
```

Training config parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Initial learning rate |
| `scheduler_step` | 2–5 | LR decay step (epochs) |
| `scheduler_gamma` | 0.7–0.9 | LR multiplicative decay factor |
| `epochs` | 20–1000 | Number of training epochs |
| `batch_size` | 1 | Batch size (graph-level) |
| `random_seed` | 42 | Reproducibility seed |

Experiments are logged to [Weights & Biases](https://wandb.ai) when `--wandb` is passed.

---

## Model Architecture

NOFE uses **KernelNN**, an edge-conditioned Graph Neural Network:

- **Lifting layer**: Projects node features from `in_width → width`
- **Message passing** (`depth` layers): Edge features drive a DenseNet kernel that generates per-edge weight matrices `[width × width]`
- **Loss**: KL divergence between Student-t affinity matrices of input data and model embeddings (local normalization)
- **Output projection**: `width → out_width`

The **subgraph variant** (KernelNN_subgraph) uses two disjoint graphs — one for observed locations and one for target locations — connected by cross-edges, enabling transductive field prediction.

Key model hyperparameters (set in model config JSON):

| Parameter | Description |
|-----------|-------------|
| `width` | Hidden embedding dimension (e.g., 8–64) |
| `ker_width` | Kernel network hidden width (e.g., 32–128) |
| `depth` | Number of message-passing layers (e.g., 3–6) |
| `in_width` | Input feature dimension |
| `out_width` | Output feature dimension |

---

## Evaluation

Evaluation runs automatically at the end of each epoch during training. To evaluate a saved checkpoint:

```bash
python src/train_standard.py \
    --config ERA5/training_configs/<config_name> \
    --eval-only \
    --checkpoint <path/to/checkpoint.pt>
```

---

## Pre-trained Models

Pre-trained model checkpoints will be released upon paper acceptance.

---

## Results

Results on ERA5 atmospheric field estimation (11 variables, test year 2019):

> Full results table will be added upon paper publication. See the paper for quantitative comparisons against baseline methods.

---

## Repository Structure

```
NOFE/
├── src/                        # Core source code
│   ├── utilities.py            # Model definitions (KernelNN, KernelNN_subgraph)
│   ├── dataset_utilities.py    # Dataset I/O and graph utilities
│   ├── generator_utility.py    # Graph construction from spatial data
│   ├── trainer/                # Training loop (base, standard, subgraph)
│   ├── train_standard.py       # Training entry point (standard / random crop)
│   ├── train_subgraph.py       # Training entry point (subgraph mode)
│   └── train_standard_nolifting.py
├── ERA5/                       # ERA5 weather application
│   ├── scripts/                # Dataset creation scripts
│   ├── configurations/         # Dataset and model config templates
│   └── training_configs/       # Experiment training configs
├── dMRI/                       # Brain dMRI application
├── modis_himalaya/             # Satellite imagery application
├── alphaEarth/                 # Satellite imagery application
├── jobs/                       # HPC job submission scripts
├── requirements2.txt           # Core Python dependencies
└── conda_environment_nofe.yml  # Full conda environment
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nofe2025,
  title     = {NOFE: Neural Operator Function Embedding},
  author    = {[Authors TBD]},
  journal   = {[Venue TBD]},
  year      = {2025},
  url       = {[URL TBD]}
}
```

---


## Acknowledgements

- ERA5 data provided by the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/)
- dMRI data provided by the [The Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release)
