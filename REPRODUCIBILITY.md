# Reproducibility Guide

All dataset creation and training scripts have been updated to ensure reproducibility through consistent random seed setting.

## Random Seed Management

### Default Seed
- Default seed value: **42**
- Used when no seed is specified in configuration files

### Setting Custom Seeds

Seeds can be configured in two places depending on the task:

#### 1. Dataset Creation
Add `random_seed` to your dataset configuration JSON file:

**ERA5 Dataset Config (`ERA5/dataset_configs/your_config.json`)**:
```json
{
    "n_points_min": 500,
    "n_points_max": 2000,
    "samples_per_date": 2,
    "k": 30,
    "perplexity": 10,
    "lifting_bool": false,
    "vars": ["w", "v", "z", "q", "r", "t"],
    "dates": ["2018-01-01", "2018-01-02"],
    "random_seed": 42
}
```

**AlphaEarth Dataset Config (`alphaEarth/dataset_configs/your_config.json`)**:
```json
{
    "n_points_min": 1000,
    "n_points_max": 5000,
    "k": 8,
    "perplexity": 30,
    "lifting_bool": false,
    "lat_min": 59.66,
    "lat_max": 59.73,
    "lon_min": 10.79,
    "lon_max": 10.86,
    "train_years": [2018, 2019, 2020],
    "test_years": [2021],
    "random_seed": 42
}
```

#### 2. Model Training
Add `random_seed` to your training configuration JSON file:

**Training Config (`{project}/training_configs/your_training.json`)**:
```json
{
    "dataset_id": "your_dataset",
    "model_id": "your_model",
    "epochs": 50,
    "batch_size": 1,
    "learning_rate": 0.0001,
    "scheduler_step": 4,
    "scheduler_gamma": 0.8,
    "random_seed": 42
}
```

## What Gets Seeded

The `set_seed()` function controls randomness across all major libraries:

1. **Python's `random` module**
   - Used for: shuffling train/test splits, random sampling

2. **NumPy's `np.random`**
   - Used for: array operations, faiss k-NN operations

3. **PyTorch's `torch`**
   - Used for: model initialization, data augmentation
   - Includes: CPU and CUDA seeds
   - Sets deterministic mode for cuDNN

## Impact on Performance

Setting `torch.backends.cudnn.deterministic = True` ensures reproducibility but may slightly reduce performance on GPU. This is a necessary trade-off for reproducible results.

## Verification

To verify reproducibility:

```bash
# Run dataset creation twice with same seed
python ERA5/scripts/create_dataset.py example_config
python ERA5/scripts/create_dataset.py example_config

# The datasets should be identical

# Run training twice with same seed
python train.py ERA5/example_training
python train.py ERA5/example_training

# Training curves and final models should be identical
```

## Technical Details

### Random Operations in Pipeline

1. **Dataset Creation**:
   - Train/test split shuffling
   - Random spatial region sampling (`sample_gridpoints_irregular`)
   - Random point sampling from regions

2. **Training**:
   - Model weight initialization
   - Dropout layers (if used)
   - Train/validation split (fallback mode)
   - Data shuffling between epochs (if implemented)

### Implementation

The seed is set at the beginning of each script using:

```python
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

This ensures all subsequent random operations use the specified seed.

## Best Practices

1. **Document your seeds**: Always note the seed used for important experiments
2. **Use the same seed for comparison**: When comparing models, use identical seeds
3. **Version your configs**: Keep dataset and training configs in version control
4. **Test reproducibility**: Periodically verify that re-running produces identical results

## Troubleshooting

### Results still not reproducible?

1. Check that CUDA operations are on the same device
2. Ensure no parallel processing is introducing non-determinism
3. Verify that all configs use the same seed value
4. Check that no random operations occur before `set_seed()` is called

### Performance degradation?

If deterministic mode is too slow for development:
- Use deterministic mode only for final/published results
- Comment out the cuDNN deterministic settings for faster iteration
- Consider using CPU for small-scale reproducibility checks
