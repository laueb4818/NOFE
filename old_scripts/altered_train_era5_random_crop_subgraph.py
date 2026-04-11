#!/usr/bin/env python3
"""
Training Script for Neural Operator Models
==========================================

Trains a graph neural network model on weather/satellite data using separate
train and test LMDB datasets.

Usage:
    python train.py <data_source>/<training_config_name>

Examples:
    python train.py ERA5/example_training
    python train.py alphaEarth/example_training
"""

import sys
import json
import os
import time
import random
import argparse
from pathlib import Path
from time import localtime, strftime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch


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

from src.utilities import (
    KernelNN_subgraph,
    GraphLMDBReader,
    setup_device,
    custom_KL_loss,
    get_output_affinities,
)


# Async data loading setup
executor = ThreadPoolExecutor(max_workers=1)

def load_batch(reader, index):
    """Helper function for async batch loading."""
    return reader[index]


def _tensor_stats_for_error(tensor: torch.Tensor) -> str:
    """Compact tensor stats for debugging errors."""
    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    device = tensor.device
    if not tensor.is_floating_point():
        return f"shape={shape}, dtype={dtype}, device={device}"
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    finite_mask = torch.isfinite(tensor)
    finite_count = finite_mask.sum().item()
    total = tensor.numel()
    if finite_count > 0:
        finite_vals = tensor[finite_mask]
        min_val = finite_vals.min().item()
        max_val = finite_vals.max().item()
        mean_val = finite_vals.mean().item()
        return (
            f"shape={shape}, dtype={dtype}, device={device}, "
            f"nan={nan_count}, inf={inf_count}, finite={finite_count}/{total}, "
            f"min={min_val:.6e}, max={max_val:.6e}, mean={mean_val:.6e}"
        )
    return (
        f"shape={shape}, dtype={dtype}, device={device}, "
        f"nan={nan_count}, inf={inf_count}, finite={finite_count}/{total}"
    )


def _assert_tensor(name: str, tensor, context: str, required: bool = True):
    if tensor is None:
        if required:
            raise RuntimeError(f"[{context}] Missing required tensor: {name}")
        return
    if not torch.is_tensor(tensor):
        raise TypeError(f"[{context}] {name} is not a torch.Tensor (got {type(tensor)})")


def _assert_dtype(name: str, tensor: torch.Tensor, expected_dtype: torch.dtype, context: str):
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"[{context}] {name} has dtype={tensor.dtype}, expected={expected_dtype}. "
            f"{_tensor_stats_for_error(tensor)}"
        )


def _assert_floating(name: str, tensor: torch.Tensor, context: str):
    if not tensor.is_floating_point():
        raise TypeError(
            f"[{context}] {name} must be floating-point. {_tensor_stats_for_error(tensor)}"
        )


def _assert_finite(name: str, tensor: torch.Tensor, context: str):
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise RuntimeError(f"[{context}] Non-finite values in {name}. {_tensor_stats_for_error(tensor)}")


def _assert_edge_index_bounds(name: str, edge_index: torch.Tensor, num_nodes: int, context: str):
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise RuntimeError(
            f"[{context}] {name} must have shape [2, E], got {tuple(edge_index.shape)}"
        )
    min_idx = edge_index.min().item()
    max_idx = edge_index.max().item()
    if min_idx < 0 or max_idx >= num_nodes:
        raise RuntimeError(
            f"[{context}] {name} index out of bounds: min={min_idx}, max={max_idx}, num_nodes={num_nodes}"
        )


def validate_subgraph_batch(batch, context: str):
    """Validate expected structure, dtype, bounds, and finiteness of a subgraph batch."""
    # Required nested data objects
    if not hasattr(batch, "source") or batch.source is None:
        raise RuntimeError(f"[{context}] Batch is missing 'source'")
    if not hasattr(batch, "target") or batch.target is None:
        raise RuntimeError(f"[{context}] Batch is missing 'target'")
    if not hasattr(batch, "cross") or batch.cross is None:
        raise RuntimeError(f"[{context}] Batch is missing 'cross'")

    # Required tensors
    _assert_tensor("source.x", batch.source.x, context)
    _assert_tensor("source.edge_index", batch.source.edge_index, context)
    _assert_tensor("source.edge_attr", batch.source.edge_attr, context)
    _assert_tensor("target.x", batch.target.x, context)
    _assert_tensor("target.edge_index", batch.target.edge_index, context)
    _assert_tensor("target.edge_attr", batch.target.edge_attr, context)
    _assert_tensor("target.input_affinities", batch.target.input_affinities, context)
    _assert_tensor("cross.edge_index", batch.cross.edge_index, context)
    _assert_tensor("cross.edges", batch.cross.edges, context)

    # Dtypes
    _assert_dtype("source.edge_index", batch.source.edge_index, torch.long, context)
    _assert_dtype("target.edge_index", batch.target.edge_index, torch.long, context)
    _assert_dtype("cross.edge_index", batch.cross.edge_index, torch.long, context)
    _assert_floating("source.x", batch.source.x, context)
    _assert_floating("source.edge_attr", batch.source.edge_attr, context)
    _assert_floating("target.x", batch.target.x, context)
    _assert_floating("target.edge_attr", batch.target.edge_attr, context)
    _assert_floating("target.input_affinities", batch.target.input_affinities, context)
    _assert_floating("cross.edges", batch.cross.edges, context)

    # Finiteness
    _assert_finite("source.x", batch.source.x, context)
    _assert_finite("source.edge_attr", batch.source.edge_attr, context)
    _assert_finite("target.x", batch.target.x, context)
    _assert_finite("target.edge_attr", batch.target.edge_attr, context)
    _assert_finite("target.input_affinities", batch.target.input_affinities, context)
    _assert_finite("cross.edges", batch.cross.edges, context)

    # Bounds checks
    source_nodes = batch.source.x.shape[0]
    target_nodes = batch.target.x.shape[0]
    _assert_edge_index_bounds("source.edge_index", batch.source.edge_index, source_nodes, context)
    _assert_edge_index_bounds("target.edge_index", batch.target.edge_index, target_nodes, context)
    _assert_edge_index_bounds("cross.edge_index", batch.cross.edge_index, max(source_nodes, target_nodes), context)

    # cross.edge_index uses source ids on row 0 and target ids on row 1
    cross_src_min = batch.cross.edge_index[0].min().item()
    cross_src_max = batch.cross.edge_index[0].max().item()
    cross_tgt_min = batch.cross.edge_index[1].min().item()
    cross_tgt_max = batch.cross.edge_index[1].max().item()
    if cross_src_min < 0 or cross_src_max >= source_nodes:
        raise RuntimeError(
            f"[{context}] cross.edge_index[0] out of source bounds: "
            f"min={cross_src_min}, max={cross_src_max}, source_nodes={source_nodes}"
        )
    if cross_tgt_min < 0 or cross_tgt_max >= target_nodes:
        raise RuntimeError(
            f"[{context}] cross.edge_index[1] out of target bounds: "
            f"min={cross_tgt_min}, max={cross_tgt_max}, target_nodes={target_nodes}"
        )

    # Shape consistency
    if batch.cross.edges.shape[0] != batch.cross.edge_index.shape[1]:
        raise RuntimeError(
            f"[{context}] cross.edges rows ({batch.cross.edges.shape[0]}) "
            f"!= cross.edge_index columns ({batch.cross.edge_index.shape[1]})"
        )
    if batch.target.edge_index.shape[1] % target_nodes != 0:
        raise RuntimeError(
            f"[{context}] target.edge_index columns ({batch.target.edge_index.shape[1]}) "
            f"not divisible by target_nodes ({target_nodes}); get_output_affinities.view may fail."
        )


def validate_forward_values(batch, out, q, loss, context: str):
    """Validate forward-pass outputs before backward/logging."""
    _assert_tensor("out", out, context)
    _assert_tensor("q", q, context)
    _assert_tensor("loss", loss, context)
    _assert_floating("out", out, context)
    _assert_floating("q", q, context)
    _assert_floating("loss", loss, context)
    _assert_finite("out", out, context)
    _assert_finite("q", q, context)
    _assert_finite("loss", loss, context)

    if out.shape[0] != batch.target.x.shape[0]:
        raise RuntimeError(
            f"[{context}] out first dim ({out.shape[0]}) != target nodes ({batch.target.x.shape[0]})"
        )
    if q.shape != batch.target.input_affinities.shape:
        raise RuntimeError(
            f"[{context}] q shape {tuple(q.shape)} != target.input_affinities shape "
            f"{tuple(batch.target.input_affinities.shape)}"
        )


def validate_model_grads(model, context: str):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(
                f"[{context}] Non-finite gradient in parameter '{name}'. "
                f"{_tensor_stats_for_error(param.grad)}"
            )


def validate_model_params(model, context: str):
    for name, param in model.named_parameters():
        if not torch.isfinite(param.data).all():
            raise RuntimeError(
                f"[{context}] Non-finite parameter values in '{name}'. "
                f"{_tensor_stats_for_error(param.data)}"
            )


def main():
    """Main training loop."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train neural operator models on weather/satellite data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python train.py ERA5/example_training\n'
               '  python train.py alphaEarth/example_training --wandb\n'
    )
    parser.add_argument('config_path', metavar='<data_source>/<training_config_name>',
                        help='Path to training configuration (e.g., ERA5/example_training)')
    parser.add_argument('--wandb', '-wb', action='store_true',
                        help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    config_path_arg = args.config_path
    use_wandb = args.wandb
    # Debugging checks are always enabled by default for this script.
    debug_checks = True
    debug_check_grads = True
    
    # Parse data source and config name
    if '/' in config_path_arg:
        data_source, training_config_id = config_path_arg.split('/', 1)
    else:
        # Backward compatibility: assume ERA5 if no slash
        print("Warning: Please specify data source (e.g., ERA5/config_name)")
        print("Assuming ERA5 for backward compatibility...\n")
        data_source = "ERA5"
        training_config_id = config_path_arg
    
    # Setup paths (relative to script directory)
    base_dir = Path(__file__).parent
    data_dir = base_dir / data_source
    
    if not data_dir.exists():
        print(f"Error: Data source directory not found: {data_dir}")
        print(f"Available directories: {[d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]}")
        sys.exit(1)
    
    training_config_path = data_dir / "configurations" / "random_crop_subgraph" / "training_configs"
    model_config_path = data_dir / "configurations" / "random_crop_subgraph" / "model_configs"
    dataset_configs_path = data_dir / "configurations" / "random_crop_subgraph" / "dataset_configs"
    dataset_path = data_dir / "datasets/random_crop_subgraph"
    
    # Output paths - project-specific directories
    trained_models_path = data_dir / "trained_models/random_crop_subgraph"
    training_logs_path = data_dir / "trained_models/random_crop_subgraph/logs"
    
    # Create output directories
    trained_models_path.mkdir(parents=True, exist_ok=True)
    (training_logs_path / "loss_records").mkdir(parents=True, exist_ok=True)
    (training_logs_path / "configs").mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device()
    print(f"Using device: {device}\n")
    
    # Load configurations
    with open(training_config_path / f"{training_config_id}.json", "r") as f:
        train_dict = json.load(f)
    
    dataset_id = train_dict['dataset_id']
    model_config_id = train_dict['model_id']
    
    with open(model_config_path / f"{model_config_id}.json", "r") as f:
        model_dict = json.load(f)
    
    with open(dataset_configs_path / f"{dataset_id}.json", "r") as f:
        dataset_dict = json.load(f)
    
    # Create model ID with timestamp
    time_string = strftime("%Y-%m-%d_t%H-%M-%S", localtime())
    model_id = f"{model_config_id}-{time_string}"
    
    # Merge all configs
    network_dict = {**train_dict, **model_dict, **dataset_dict}
    
    # Set random seed for reproducibility
    seed = train_dict.get('random_seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Extract training parameters
    learning_rate = train_dict['learning_rate']
    scheduler_step = train_dict['scheduler_step']
    scheduler_gamma = train_dict['scheduler_gamma']
    epochs = train_dict['epochs']
    
    print(f"Data source: {data_source}")
    print(f"Configuration: {training_config_id}")
    print(f"Model ID: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {data_source}/trained_models/")
    print(f"Weights & Biases: {'enabled' if use_wandb else 'disabled'}\n")
    print("Debug checks: enabled (gradient checks: enabled)\n")
    
    # Initialize Weights & Biases (optional)
    if use_wandb:
        import wandb
        wandb.init(
            project=f"NOFE-{data_source.lower()}",
            config=network_dict,
            name=model_id,
            settings=wandb.Settings(code_dir=".")
        )
    
    # Load datasets (separate train and test)
    train_dataset_path = dataset_path / f"{dataset_id}_train.lmdb"
    test_dataset_path = dataset_path / f"{dataset_id}_test.lmdb"
    
    if not train_dataset_path.exists():
        print(f"Error: Training dataset not found: {train_dataset_path}")
        sys.exit(1)
    
    if not test_dataset_path.exists():
        print(f"Warning: Test dataset not found: {test_dataset_path}")
        print("Continuing with train dataset only...")
        test_reader = None
    else:
        test_reader = GraphLMDBReader(str(test_dataset_path))
        print(f"Test dataset: {len(test_reader)} samples")
    
    train_reader = GraphLMDBReader(str(train_dataset_path))
    print(f"Train dataset: {len(train_reader)} samples")
    
    # Use all training data (no manual split)
    train_indices = list(range(len(train_reader)))
    
    # Use test dataset as validation set
    if test_reader is not None:
        val_indices = list(range(len(test_reader)))
        val_reader = test_reader
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples (from test set): {len(val_indices)}\n")
    else:
        # Fallback: if no test dataset, split train data
        print("Warning: No test dataset found, falling back to train/val split (90/10)")
        train_count = len(train_reader)
        train_size = int(0.9 * train_count)
        indices = list(range(train_count))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        val_reader = train_reader
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}\n")
    
    # Create model
    # Support both old (single node_features) and new (separate input/output) configs
    # For backward compatibility, fall back to 'node_features' if separate dimensions not specified
    input_node_dim = model_dict.get('input_node_features', model_dict.get('node_features', 1))
    output_node_dim = model_dict.get('output_node_features', model_dict.get('node_features', 1))
    
    model = KernelNN_subgraph(
        width=model_dict['width'],
        ker_width=model_dict['ker_width'],
        depth=model_dict['depth'],
        ker_in=model_dict['edge_attr'],
        known_node_dim=input_node_dim,  # input feature dimension
        unknown_node_dim=output_node_dim,  # output feature dimension
        out_width=model_dict['out_width'],
    ).to(device)
    
    print(f"Model: KernelNN")
    print(f"  Width: {model_dict['width']}")
    print(f"  Depth: {model_dict['depth']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    
    # Training records
    loss_record = []
    val_loss_record = []
    epoch_record = []
    time_record = []
    lr_record = []
    
    # Training loop
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Epoch 0: Initial validation (before training)
    print("Epoch 0: Evaluating initial model...")
    model.eval()
    initial_val_losses = []
    if debug_checks:
        validate_model_params(model, context="epoch=0 pre-validation model params")
    
    for j, val_ind in enumerate(val_indices):
        print(f'Epoch 0/{epochs} - Validation: {j+1}/{len(val_indices)}', end='\r')
        batch = load_batch(val_reader, val_ind)
        batch = batch.to(device)
        if debug_checks:
            validate_subgraph_batch(batch, context=f"phase=val_init epoch=0 sample_idx={val_ind} loop_j={j}")
        with torch.no_grad():
            out = model(batch)
            q = get_output_affinities(out, batch.target.edge_index)
            loss = custom_KL_loss(batch.target.input_affinities, q).mean()
            if debug_checks:
                validate_forward_values(
                    batch, out, q, loss,
                    context=f"phase=val_init epoch=0 sample_idx={val_ind} loop_j={j}"
                )
            initial_val_losses.append(loss.item())
    
    initial_val_loss = np.mean(initial_val_losses)
    loss_record.append(np.nan)  # No training loss for epoch 0
    val_loss_record.append(initial_val_loss)
    epoch_record.append(0)
    lr_record.append(learning_rate)
    time_record.append("00:00")
    
    if use_wandb:
        wandb.log({
            'epoch': 0,
            'ep_val_loss': initial_val_loss,
        })
    
    print(f'\nVal Loss: {round(initial_val_loss, 4)} (before training)\n')
    
    future = None
    
    for ep in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        all_losses = []
        
        # Training
        for i, train_ind in enumerate(train_indices):

            print(f"\r\033[KEpoch {ep}/{epochs} - Training: {i+1}/{len(train_indices)}", end="", flush=True)
            
            # Prefetch next batch
            if i + 1 < len(train_indices):
                future = executor.submit(load_batch, train_reader, train_indices[i + 1])
            
            # Load current batch
            if i == 0:
                batch = load_batch(train_reader, train_ind)
            else:
                batch = future.result()
            
            batch = batch.to(device)
            if debug_checks:
                validate_subgraph_batch(batch, context=f"phase=train epoch={ep} sample_idx={train_ind} loop_i={i}")
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch)
            q = get_output_affinities(out, batch.target.edge_index)
            loss = custom_KL_loss(batch.target.input_affinities, q).mean()
            if debug_checks:
                validate_forward_values(
                    batch, out, q, loss,
                    context=f"phase=train epoch={ep} sample_idx={train_ind} loop_i={i}"
                )
            all_losses.append(loss.item())
            
            if use_wandb:
                wandb.log({'loss': loss.item()})
            
            # Backward pass
            loss.backward()
            if debug_checks and debug_check_grads:
                validate_model_grads(model, context=f"phase=train epoch={ep} sample_idx={train_ind} loop_i={i} post-backward")
            optimizer.step()
            if debug_checks and debug_check_grads:
                validate_model_params(model, context=f"phase=train epoch={ep} sample_idx={train_ind} loop_i={i} post-step")
        
        avg_loss = np.mean(all_losses) + 10e-8
        loss_record.append(avg_loss)
        epoch_record.append(ep)
        lr_record.append(scheduler.get_last_lr()[0])
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        future_val = None
        
        for j, val_ind in enumerate(val_indices):
            print(f"\r\033[KEpoch {ep}/{epochs} - Training: {i+1}/{len(train_indices)} - Validation: {j+1}/{len(val_indices)}", end="", flush=True)
            # Prefetch next batch
            if j + 1 < len(val_indices):
                future_val = executor.submit(load_batch, val_reader, val_indices[j + 1])
            
            # Load current batch
            if j == 0:
                batch = load_batch(val_reader, val_ind)
            else:
                batch = future_val.result()
            
            batch = batch.to(device)
            if debug_checks:
                validate_subgraph_batch(batch, context=f"phase=val epoch={ep} sample_idx={val_ind} loop_j={j}")
            
            with torch.no_grad():
                out = model(batch)
                q = get_output_affinities(out, batch.target.edge_index)
                loss = custom_KL_loss(batch.target.input_affinities, q).mean()
                if debug_checks:
                    validate_forward_values(
                        batch, out, q, loss,
                        context=f"phase=val epoch={ep} sample_idx={val_ind} loop_j={j}"
                    )
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        val_loss_record.append(val_loss)
        
        # Epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        minutes, seconds = divmod(int(epoch_duration), 60)
        time_str = f"{minutes:02}:{seconds:02}"
        time_record.append(time_str)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': ep,
                'ep_loss': avg_loss,
                'ep_val_loss': val_loss,
            })
        
        print(" " * 120, end="\r")
        print(f'Loss: {round(avg_loss, 4)}\t|  Val Loss: {round(val_loss, 4)}\t|  LR: {round(scheduler.get_last_lr()[0], 7)}\t|  Time: {time_str}')
        
        # Save checkpoint every 5 epochs
        if ep % 5 == 0:
            loss_df = pd.DataFrame({
                'epoch': epoch_record,
                'loss': loss_record,
                'val_loss': val_loss_record,
                'lr': lr_record,
                'time': time_record
            })
            
            torch.save(model.state_dict(), trained_models_path / f"{model_id}.pth")
            loss_df.to_csv(training_logs_path / "loss_records" / f"{model_id}.csv")
            
            with open(training_logs_path / "configs" / f"{model_id}.json", 'w') as f:
                json.dump(network_dict, f, indent=4)
            
            print(f'✓ Checkpoint saved: {model_id}\n')
    
    # Final save
    loss_df = pd.DataFrame({
        'epoch': epoch_record,
        'loss': loss_record,
        'val_loss': val_loss_record,
        'lr': lr_record,
        'time': time_record
    })
    
    torch.save(model.state_dict(), trained_models_path / f"{model_id}.pth")
    loss_df.to_csv(training_logs_path / "loss_records" / f"{model_id}.csv")
    
    with open(training_logs_path / "configs" / f"{model_id}.json", 'w') as f:
        json.dump(network_dict, f, indent=4)
    
    print("\n" + "="*60)
    print(f'✓ Training complete!')
    print(f'Model saved: {model_id}')
    print("="*60)
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
