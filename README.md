# NOFE: Neural Operator Function Embedding

Official implementation of **NOFE — Neural Operator Function Embedding**
([arXiv:2605.11970](https://arxiv.org/abs/2605.11970) · [PDF](https://arxiv.org/pdf/2605.11970v2)).

> Lars Uebbing, Harald L. Joakimsen, Siyan Chen, Georgios Leontidis, Kristoffer K. Wickstrøm,
> Michael C. Kampffmeyer, Sébastien Lefèvre, Arnt-Børre Salberg, Robert Jenssen

Most dimensionality-reduction methods treat data as discrete point clouds and ignore the continuous
domain structure of the underlying process. NOFE is a domain-aware framework for continuous
dimensionality reduction: it learns a **function-to-function mapping** with a Graph Kernel Operator, so a
trained model can be evaluated, mesh-free, at arbitrary query locations independent of the input
discretization. NOFE is derived as an approximation of sheaf-to-sheaf mappings, generalizing Sheaf Neural
Networks to continuous domains.

This repository ships the four applications from the paper — ERA5 climate reanalysis, dMRI brain volumes,
MODIS Himalaya satellite tiles and AlphaEarth embeddings — with dataset creation, all training modes, the
evaluation scripts and the figure notebooks.

```bash
git clone https://github.com/laueb4818/NOFE.git
cd NOFE
```

**Contents:** [Results](#results-at-a-glance) · [Installation](#installation) · [Data](#data) ·
[Dataset creation](#dataset-creation) · [Training](#training) · [Model architecture](#model-architecture) ·
[Configurations](#configurations) · [Evaluation](#evaluation-and-figures) · [Notebooks](#notebooks) ·
[Pre-trained models](#pre-trained-models) · [Repository structure](#repository-structure) ·
[Citation](#citation) · [License](#license)

---

## Results at a glance

Metrics reported in the paper for the ERA5 climate reanalysis dataset:

| Metric | NOFE | PCA | t-SNE | UMAP |
|---|---|---|---|---|
| Local Stress (lower is better) | **0.111** | 0.398 | 0.773 | 0.791 |
| Stress-1, global structure (lower is better) | 0.379 | **0.268** | — | — |
| Patch Stitching Error, regional normalization (lower is better) | **59.0** | — | — | 267.6 |

NOFE preserves local structure substantially better than all baselines, reduces the Patch Stitching Error
by up to **20.0×** relative to UMAP, and stays consistent across disjoint domain patches, while remaining
competitive on global structure. The `ERA5/evaluation_springfield/*_scores/` and `*_errors/` JSONs in this
repository are the recorded metrics behind these numbers.

---

## Installation

Conda (recommended):

```bash
conda env create -f conda_environment_nofe.yml
conda activate nofe
```

pip alternative:

```bash
pip install -r requirements2.txt
```

`requirements.txt` is the CUDA-2.2-pinned alternative (`torch==2.2.0+cu118` with the matching
`torch-scatter` index). `env.yaml` and `env_modified.yaml` are macOS-arm64 reference exports of the
environment used for the paper runs; they are not the primary instructions.

Key dependencies: Python 3.10, PyTorch ≥ 2.0, [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
2.6.1, `torch-scatter`, `faiss-cpu` (k-NN graph construction), `xarray`/`netcdf4` (ERA5), `nibabel`/`dipy`
(dMRI), `rasterio` (satellite imagery), `wandb` (optional).

---

## Data

Nothing under `**/data/` or `**/datasets/` is committed — both are gitignored, along with
`**/trained_models/`, `**/wandb/` and the usual model/array file extensions.

| Application | Download | Place in |
|---|---|---|
| ERA5 | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/), one NetCDF file per day named `YYYY-MM-DD.nc` | `ERA5/data/` |
| dMRI | [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult), 7T diffusion | `dMRI/data/<subject>/` |
| AlphaEarth | Google/AlphaEarth annual embedding GeoTIFFs, `sat_emb_<year>.tif` | `alphaEarth/data/` |
| MODIS Himalaya | MODIS brightness-temperature tiles | `modis_himalaya/data/himalaya/` |

`ERA5/data/` may be a symlink to an external directory; the ignore rule covers the link itself.

---

## Dataset creation

Each script takes one positional argument, the config id, and resolves it to
`<app>/configurations/<mode>/dataset_configs/<id>.json`. Outputs are written to the `dataset_save_path`
recorded in that config (`<app>/datasets/<mode>/`). The scripts `chdir` into their own directory, so they
may be invoked from anywhere.

ERA5:

```bash
cd ERA5/scripts && python create_dataset.py toyset_reg_fixcrop_ones_woPos
cd ERA5/scripts && python create_dataset_random_crop.py dataset1_newSigma
cd ERA5/scripts && python create_subgraph_dataset.py toyset_sub_fixcrop_median_woPos
cd ERA5/scripts && python create_dataset_random_crop_subgraph.py baseline
```

dMRI:

```bash
cd dMRI/scripts && python create_dataset.py base
cd dMRI/scripts && python create_subgraph_dataset.py base
```

MODIS Himalaya:

```bash
cd modis_himalaya/scripts && python create_dataset.py base
cd modis_himalaya/scripts && python create_dataset_random_crop.py base
cd modis_himalaya/scripts && python create_subgraph_dataset.py dataset_subgraph
cd modis_himalaya/scripts && python create_dataset_random_crop_subgraph.py base
```

AlphaEarth:

```bash
cd alphaEarth/scripts && python create_dataset.py base
cd alphaEarth/scripts && python create_subgraph_dataset.py subgraph_base
```

Each run also writes `<id>_config.json` next to the LMDB files; the trainer reads it back, so a dataset
carries the graph-construction settings it was built with.

---

## Training

Three entry points, one per mode:

| Mode | Entry point |
|---|---|
| standard (fixed region) | `src/train_standard.py` |
| standard without lifting layer | `src/train_standard_nolifting.py` |
| subgraph (disjoint input/output graphs) | `src/train_subgraph.py` |

The positional argument is `<project>/<config_name>` and resolves to
`<project>/training_configs/<config_name>.json` — do not include `training_configs/` in the argument.

```bash
python src/train_standard.py ERA5/regular_base
python src/train_standard.py ERA5/regular_papermodel1
python src/train_standard.py ERA5/random_crop_base
python src/train_subgraph.py ERA5/subgraph_base
python src/train_standard_nolifting.py ERA5/regular_papermodel4b_nolifting
python src/train_standard.py dMRI/base
python src/train_subgraph.py dMRI/subgraph_base
python src/train_standard.py modis_himalaya/testtraining2
python src/train_standard.py alphaEarth/base
python src/train_subgraph.py alphaEarth/subgraph_base
```

Flags:

```
--wandb / -wb                            enable Weights & Biases logging
--gradient-clipping / --no-gradient-clipping
--track-gradients / --no-track-gradients
--checkpoint-every N                     checkpoint interval (default 5)
```

Flags override the values in the training config. Checkpoints land in
`<app>/trained_models/<mode>/`, loss records and resolved configs under `<app>/trained_models/<mode>/logs/`.

`train_standard_nolifting.py` has no lifting layer, so the model config's `node_features` must equal
`width`.

---

## Model architecture

NOFE uses **KernelNN**, an edge-conditioned graph neural network:

- **Lifting layer**: projects node features `node_features → width` (omitted in the no-lifting variant)
- **Message passing** (`depth` layers): edge attributes drive a DenseNet kernel that emits a per-edge
  `[width × width]` weight matrix (an `NNConv`-style convolution)
- **Loss**: KL divergence between the Student-t affinity matrices of the input features and of the model
  output
- **Output projection**: `width → out_width`

Affinities are normalized **locally**, i.e. per node over its own k neighbours. `get_output_affinities` in
`src/utilities.py` is aliased to `get_output_affinities_local_norm`: global normalization (dividing by the
sum over all node pairs) produces exploding gradients and unstable training, so it is not used.

The **subgraph variant** (`KernelNN_subgraph`) keeps two disjoint graphs — observed (`source`) and query
(`target`) — joined by a cross graph, and applies a separate kernel to the cross edges. That is what allows
prediction on a different point set than the input, and it is the mode the sampling-independence and
patch-stitching experiments use.

| Parameter | Meaning |
|---|---|
| `width` | hidden embedding dimension |
| `ker_width` | kernel-network hidden width |
| `depth` | number of message-passing layers |
| `node_features` / `in_width` | input feature dimension |
| `edge_attr` | edge feature dimension |
| `out_width` | output embedding dimension |

---

## Configurations

Dataset and model configs live under `<app>/configurations/<mode>/{dataset,model}_configs/`; training
configs live under `<app>/training_configs/`. The modes are `regular`, `random_crop`, `subgraph` and
`random_crop_subgraph`.

The eight regular-mode test models sweep width, kernel width and depth (this is the parameter sweep the
paper's ablations use):

| `width` | `ker_width` | `depth` |
|---|---|---|
| 8 | 32 | 3 |
| 8 | 32 | 6 |
| 8 | 128 | 3 |
| 8 | 128 | 6 |
| 64 | 32 | 3 |
| 64 | 32 | 6 |
| 64 | 128 | 3 |
| 64 | 128 | 6 |

The subgraph mode ships the same sweep under `ERA5/configurations/subgraph/model_configs/testmodel1…10`
(width 8→128, kernel width 32→256, depth 1→4) plus `base_d3`/`base2_d2`, which are what `subgraph_base` and
`subgraph_tkernel_*` use. The paper's headline models are the `papermodel*` configs.

ERA5 dataset-config names follow `toyset_reg_[fix|rand]crop_[var|perp|ones]_[w|wo]Pos`: `fix`/`rand`
selects a fixed or random bounding box, `var`/`perp`/`ones` records the sigma setting the config was
originally written for, and `wPos`/`woPos` selects whether node positions are part of the node features
(`channel_and_position` vs `channel_only`).

Note that neither `sigma_method` nor `perplexity` affects graph construction: the k-NN graph is selected
purely by `k`, and input affinities are computed from the node features. `perplexity` is still accepted in
dataset configs for compatibility.

---

## Evaluation and figures

The four evaluation scripts each take a trained-model id and write a JSON of metrics into the directory
next to the script:

```bash
cd ERA5/evaluation_springfield
python distance_preservation.py regular_papermodel1_2026-05-04_t11-31-09   # -> distance_preservation_scores/
python gluing_error.py  regular_papermodel1_2026-05-04_t11-31-09           # -> gluing_errors/
python patch_stitching_new.py regular_papermodel1_2026-05-04_t11-31-09     # -> patch_stitching_errors/
python patch_stitching_v2_new.py regular_papermodel1_2026-05-04_t11-31-09  # -> patch_stitching_errors_v2/
```

`ERA5/scripts/eval_computation_cost.py` measures per-sample preprocessing/embedding time and writes
`ERA5/scripts/computation_times*.csv`; those CSVs are the paper's timing table and are committed. It needs
`ERA5/scripts/graphs.pt`, which is **not** shipped — the script's default preprocessing pass rebuilds it
from the LMDB datasets (`--no-preprocessing` skips that step).

Baseline embeddings live one directory per method, and each runs in its own conda env:

| Method | Directory | Entry point |
|---|---|---|
| UMAP | `ERA5/umap/` | `execute_umap.py` |
| Concord | `ERA5/concord/` | `execute_concord.py` (`environment_concord.yml`) |
| SpatialPCA | `ERA5/spatialpca/` | `execute_spatialpca_server.R`, `execute_spatialpca_single.R` |

`execute_umap.py` and `execute_concord.py` are also driven as subprocesses from the ERA5, dMRI and
modis_himalaya notebooks, so `ERA5/umap/` is the single copy of the UMAP runner.

SpatialPCA needs both environments:

```bash
conda env create -f spatialpca_environment.yml
bash ERA5/spatialpca/install_spatialpca_R_deps.sh
```

`ERA5/paper_figures/lipschitz-stat.ipynb` shells out to `Rscript` via the `SPATIALPCA_RSCRIPT` environment
variable (default `Rscript`, i.e. whatever is on `PATH`):

```bash
export SPATIALPCA_RSCRIPT="$HOME/miniforge3/envs/spatialpca/bin/Rscript"
```

---

## Notebooks

All notebooks are stored without outputs. Run them from their own directory: every notebook sits at
`<app>/<subdir>/<name>.ipynb`, so it resolves the repository root as `Path.cwd().parent.parent`.

**ERA5 — dataset and predictions**

- `ERA5/notebooks/data_viewer.ipynb` — raw ERA5 NetCDF viewer (lat/lon maps of single variables)
- `ERA5/notebooks/visualize.ipynb` — standard-mode prediction viewer
- `ERA5/notebooks/visualize_rand_crop_subgraph.ipynb` — random-crop subgraph prediction viewer
- `ERA5/notebooks/distance_preservation.ipynb` — distance-preservation scatter/error analysis
- `ERA5/notebooks/eval_computation_cost.ipynb` — reads the committed `computation_times*.csv` and plots the cost table

**ERA5 — paper figures**

- `ERA5/paper_figures/2d_scatter.ipynb` — 2-D embeddings: NOFE, PCA, t-SNE, UMAP, PHATE, SpatialPCA, Concord
- `ERA5/paper_figures/grayscale_correlation.ipynb` — 3-D → grayscale projection and per-variable correlation
- `ERA5/paper_figures/lipschitz-stat.ipynb` — Lipschitz-constant distributions over model configurations
- `ERA5/paper_figures/patch_stitching.ipynb` — stitched-patch reconstruction from independent crops
- `ERA5/paper_figures/patch_stitching_withOverlap.ipynb` — overlap and gluing behaviour between patches
- `ERA5/paper_figures/patch_stitching_procrustes_quantitative.ipynb` — Procrustes alignment of patch embeddings (quantitative)
- `ERA5/paper_figures/distance_preservation_quantitative.ipynb` — distance-preservation metrics over all baselines (quantitative)
- `ERA5/paper_figures/distance_preservation_super_resolution.ipynb` — super-resolution / query-density study
- `ERA5/paper_figures/visualize_subgraph.ipynb` — subgraph-mode predictions against all baselines
- `ERA5/evaluation_springfield/patch_stitching_withOverlap_quantitative.ipynb` — overlap-error metrics with a KD-tree pairing

**dMRI**

- `dMRI/notebooks/check_images.ipynb` — HCP b0 / mask quality check
- `dMRI/notebooks/view_results.ipynb` — standard-mode prediction viewer
- `dMRI/notebooks/view_subgraph_results.ipynb` — subgraph-mode prediction viewer
- `dMRI/notebooks/dMRI_patch_stitching.ipynb` — patch stitching on brain volumes
- `dMRI/notebooks/dMRI_patch_gluing_withOverlap_quantitative.ipynb` — gluing error with overlapping patches
- `dMRI/notebooks/dMRI_distance_preservation.ipynb` — distance-preservation figures for dMRI

**MODIS Himalaya**

- `modis_himalaya/notebooks/view_data.ipynb` — tile viewer
- `modis_himalaya/notebooks/visualize.ipynb` — prediction viewer
- `modis_himalaya/notebooks/patch_stitching.ipynb` — patch stitching on satellite tiles
- `modis_himalaya/notebooks/patch_stitching_with_Overlap.ipynb` — overlap-error analysis
- `modis_himalaya/notebooks/distance_preservation.ipynb` — distance-preservation figures

**AlphaEarth**

- `alphaEarth/notebooks/visualize_subgraph.ipynb` — subgraph-mode predictions on AlphaEarth embeddings

---

## Pre-trained models

Model checkpoints are **not** released — data, trained weights and run outputs are all gitignored. The
recorded metrics from the paper runs are committed, so the quantitative results remain checkable without
the checkpoints:

- `ERA5/evaluation_springfield/distance_preservation_scores/` — distance-preservation scores
- `ERA5/evaluation_springfield/gluing_errors/` — gluing errors
- `ERA5/evaluation_springfield/patch_stitching_errors/`, `patch_stitching_errors_v2/` — patch-stitching errors
- `ERA5/scripts/computation_times*.csv` — preprocessing/embedding timings

Notebooks that consume trained models read them from `<app>/trained_models/<mode>/<model_id>.pth` plus the
matching `<model_id>.json` under `logs/configs/`; train first, then run the notebook.

---

## Repository structure

```
NOFE/
├── src/
│   ├── utilities.py              # KernelNN, KernelNN_subgraph, losses, LMDB reader/writer
│   ├── generator_utility.py      # graph construction from spatial data
│   ├── plot_utilities.py         # shared plotting helpers
│   ├── trainer/                  # base + standard + no-lifting + subgraph trainers
│   ├── train_standard.py
│   ├── train_standard_nolifting.py
│   └── train_subgraph.py
├── ERA5/
│   ├── scripts/                  # dataset creation + computation-cost evaluation
│   ├── configurations/           # dataset/model configs per mode
│   ├── training_configs/         # training configs
│   ├── evaluation_springfield/   # evaluation scripts, recorded metrics, one notebook
│   ├── paper_figures/            # figure notebooks
│   ├── notebooks/                # dataset/prediction viewers, cost table
│   ├── umap/  concord/  spatialpca/   # baseline embedding runners
│   └── data/  datasets/  trained_models/   # gitignored
├── dMRI/                         # same layout, dMRI application
├── modis_himalaya/               # same layout, MODIS application
├── alphaEarth/                   # same layout, AlphaEarth application
├── Dockerfile
├── LICENSE
├── conda_environment_nofe.yml
├── env.yaml  env_modified.yaml   # macOS-arm64 reference exports
├── requirements.txt  requirements2.txt
└── spatialpca_environment.yml
```

---

## Citation

If you use this code or the NOFE method in your research, please cite:

```bibtex
@misc{uebbing2026nofe,
  title         = {NOFE -- Neural Operator Function Embedding},
  author        = {Lars Uebbing and Harald L. Joakimsen and Siyan Chen and Georgios Leontidis
                   and Kristoffer K. Wickstr{\o}m and Michael C. Kampffmeyer and S{\'e}bastien Lef{\`e}vre
                   and Arnt-B{\o}rre Salberg and Robert Jenssen},
  year          = {2026},
  eprint        = {2605.11970},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.11970}
}
```

---

## License

Released under the [MIT License](LICENSE) — © 2026 the NOFE authors.

The license covers the source code in this repository. The datasets it operates on keep their own terms:
ERA5 data is subject to the Copernicus Climate Change Service licence, HCP dMRI data to the Human
Connectome Project open-access terms, and the AlphaEarth and MODIS products to their respective providers'
terms. Trained model checkpoints are not distributed.

---

## Acknowledgements

- ERA5 data provided by the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/)
- dMRI data provided by the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release)
