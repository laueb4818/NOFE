"""
Base Trainer for Neural Operator Models.

Provides shared training infrastructure including:
- Configuration loading from JSON file
- Device setup and seed management
- Data loading (train/val readers)
- Optimizer and scheduler setup
- Training loop with validation
- Checkpoint and log saving
- Weights & Biases integration

CLI Arguments:
    --config <project>/<config_name>   Config file path (e.g., ERA5/example_training)
    --wandb                             Enable Weights & Biases logging
    --gradient-clipping / --no-gradient-clipping
    --track-gradients / --no-track-gradients
    --checkpoint-every <N>

Config File Structure:
    {
        "model_id": "model_1",
        "model_config_path": "ERA5/configurations/regular/model_configs/model_1.json",
        "dataset_id": "dataset_1",
        "dataset_path": "ERA5/datasets/regular",
        "learning_rate": 0.001,
        "scheduler_step": 10,
        "scheduler_gamma": 0.5,
        "epochs": 100,
        "random_seed": 42,
        "use_gradient_clipping": false,
        "track_gradients": false,
        "checkpoint_every": 5
    }
    
    Note: The dataset config is automatically loaded from {dataset_path}/{dataset_id}_config.json
    which is created during dataset generation.
"""

import sys
import json
import random
import argparse
from pathlib import Path
from time import localtime, strftime
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from src.utilities import (
    GraphLMDBReader,
    setup_device,
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseTrainer(ABC):
    """Shared training logic for standard and subgraph trainers."""
    
    def __init__(
        self,
        use_gradient_clipping=None,
        track_gradients=None,
        checkpoint_every=None,
    ):
        """
        Initialize trainer.
        
        Args:
            use_gradient_clipping: Apply gradient clipping after backward pass (default: False)
            track_gradients: Track and log gradient norms (default: False)
            checkpoint_every: Save checkpoint every N epochs (default: 5)
        """
        self.use_gradient_clipping_cli = use_gradient_clipping
        self.track_gradients_cli = track_gradients
        self.checkpoint_every_cli = checkpoint_every
        
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_reader = None
        self.val_reader = None
        self.train_indices = None
        self.val_indices = None
        self.network_dict = None
        self.model_id = None
        self.use_wandb = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        self.config = None
    
    @abstractmethod
    def create_model(self, model_dict):
        """Create and return the model instance."""
        pass
    
    @abstractmethod
    def forward_step(self, batch):
        """Run forward pass and compute loss for a single batch.
        
        Returns:
            tuple: (output, loss_tensor)
        """
        pass
    
    def run(self, argv=None):
        """Main entry point for training."""
        args = self._parse_args(argv)
        
        config_path = self._resolve_config_path(args.config)
        self.config = self._load_config(config_path)
        
        self._apply_cli_overrides(args)
        
        self.use_wandb = args.wandb
        
        project_root = config_path.parent.parent.parent
        model_config_path = self._resolve_path(self.config['model_config_path'], base_dir=project_root)
        
        # Derive model_id from model_config_path filename
        model_id = model_config_path.stem
        
        dataset_path = self._resolve_path(self.config['dataset_path'], base_dir=project_root)
        dataset_id = self.config['dataset_id']
        
        dataset_config_path = dataset_path / f"{dataset_id}_config.json"
        
        trained_models_path = dataset_path.parent.parent / "trained_models"
        training_logs_path = trained_models_path / "logs"
        
        trained_models_path.mkdir(parents=True, exist_ok=True)
        (training_logs_path / "loss_records").mkdir(parents=True, exist_ok=True)
        (training_logs_path / "configs").mkdir(parents=True, exist_ok=True)
        
        self.device = setup_device()
        print(f"Using device: {self.device}\n")
        
        with open(model_config_path, "r") as f:
            model_dict = json.load(f)
        
        with open(dataset_config_path, "r") as f:
            dataset_dict = json.load(f)
        
        time_string = strftime("%Y-%m-%d_t%H-%M-%S", localtime())
        self.model_id = f"{model_id}-{time_string}"
        
        self.network_dict = {**self.config, **model_dict, **dataset_dict}
        
        seed = self.config.get('random_seed', 42)
        set_seed(seed)
        print(f"Random seed: {seed}")
        
        learning_rate = self.config['learning_rate']
        scheduler_step = self.config['scheduler_step']
        scheduler_gamma = self.config['scheduler_gamma']
        epochs = self.config['epochs']
        use_gradient_clipping = self.config['use_gradient_clipping']
        track_gradients = self.config['track_gradients']
        checkpoint_every = self.config['checkpoint_every']
        
        print(f"Configuration: {args.config}")
        print(f"Model ID: {self.model_id}")
        print(f"Dataset ID: {dataset_id}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Gradient clipping: {'enabled' if use_gradient_clipping else 'disabled'}")
        print(f"Track gradients: {'enabled' if track_gradients else 'disabled'}")
        print(f"Checkpoint every: {checkpoint_every} epochs")
        print(f"Output directory: {trained_models_path}")
        print(f"Weights & Biases: {'enabled' if self.use_wandb else 'disabled'}\n")
        
        if self.use_wandb:
            import wandb
            wandb.init(
                project=f"NOFE-{dataset_path.parent.name}",
                config=self.network_dict,
                name=self.model_id,
                settings=wandb.Settings(code_dir=".")
            )
        
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
        
        self.train_reader = GraphLMDBReader(str(train_dataset_path))
        print(f"Train dataset: {len(self.train_reader)} samples")
        
        self.train_indices = list(range(len(self.train_reader)))
        
        if test_reader is not None:
            self.val_indices = list(range(len(test_reader)))
            self.val_reader = test_reader
            print(f"Train samples: {len(self.train_indices)}")
            print(f"Val samples (from test set): {len(self.val_indices)}\n")
        else:
            print("Warning: No test dataset found, falling back to train/val split (90/10)")
            train_count = len(self.train_reader)
            train_size = int(0.9 * train_count)
            indices = list(range(train_count))
            random.shuffle(indices)
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size:]
            self.val_reader = self.train_reader
            print(f"Train samples: {len(self.train_indices)}")
            print(f"Val samples: {len(self.val_indices)}\n")
        
        self.model = self.create_model(model_dict)
        self.model.to(self.device)
        
        print(f"Model: {self.model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters())}\n")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=5e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )
        
        self._train_loop(epochs, trained_models_path, training_logs_path, 
                         use_gradient_clipping, track_gradients, checkpoint_every)
        
        if self.use_wandb:
            import wandb
            wandb.finish()
    
    def _resolve_config_path(self, config_arg):
        """Resolve config path from CLI argument.
        
        Args:
            config_arg: <project>/<config_name> format
            
        Returns:
            Path to config JSON file
        """
        parts = config_arg.rsplit('/', 1)
        if len(parts) == 1:
            raise ValueError(f"Invalid config format: {config_arg}. Expected <project>/<config_name>")
        
        project, config_name = parts
        
        trainer_dir = Path(__file__).parent
        base_dir = trainer_dir.parent.parent
        config_path = base_dir / project / "training_configs" / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        return config_path
    
    def _resolve_path(self, path_str, base_dir=None):
        """Resolve path relative to trainer script location or specified base."""
        if base_dir is None:
            trainer_dir = Path(__file__).parent
            base_dir = trainer_dir.parent.parent
        resolved = base_dir / path_str
        return resolved.resolve()
    
    def _load_config(self, config_path):
        """Load config from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_fields = [
            'model_config_path', 'dataset_id', 
            'dataset_path', 'learning_rate',
            'scheduler_step', 'scheduler_gamma', 'epochs'
        ]
        missing = [f for f in required_fields if f not in config]
        if missing:
            raise ValueError(f"Missing required fields in config: {missing}")
        
        config.setdefault('random_seed', 42)
        config.setdefault('use_gradient_clipping', False)
        config.setdefault('track_gradients', False)
        config.setdefault('checkpoint_every', 5)
        
        return config
    
    def _apply_cli_overrides(self, args):
        """Apply CLI argument overrides to config."""
        if self.use_gradient_clipping_cli is not None:
            self.config['use_gradient_clipping'] = self.use_gradient_clipping_cli
        
        if self.track_gradients_cli is not None:
            self.config['track_gradients'] = self.track_gradients_cli
        
        if self.checkpoint_every_cli is not None:
            self.config['checkpoint_every'] = self.checkpoint_every_cli
    
    def _train_loop(self, epochs, trained_models_path, training_logs_path,
                    use_gradient_clipping, track_gradients, checkpoint_every):
        """Execute the training loop."""
        import time
        
        loss_record = []
        val_loss_record = []
        epoch_record = []
        time_record = []
        lr_record = []
        
        print("=" * 60)
        print("Starting training...")
        print("=" * 60 + "\n")
        
        print("Epoch 0: Evaluating initial model...")
        self.model.eval()
        initial_val_losses = []
        
        for j, val_ind in enumerate(self.val_indices):
            print(f'Epoch 0/{epochs} - Validation: {j+1}/{len(self.val_indices)}', end='\r')
            batch = self.train_reader[val_ind].to(self.device)
            
            with torch.no_grad():
                _, loss = self.forward_step(batch)
                initial_val_losses.append(loss.item())
        
        initial_val_loss = np.mean(initial_val_losses)
        loss_record.append(np.nan)
        val_loss_record.append(initial_val_loss)
        epoch_record.append(0)
        lr_record.append(self.scheduler.get_last_lr()[0])
        time_record.append("00:00")
        
        if self.use_wandb:
            import wandb
            wandb.log({'epoch': 0, 'ep_val_loss': initial_val_loss})
        
        print(f'\nVal Loss: {round(initial_val_loss, 4)} (before training)\n')
        
        future = None
        
        for ep in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            all_losses = []
            gns = []
            
            for i, train_ind in enumerate(self.train_indices):
                print(f"\r\033[KEpoch {ep}/{epochs} - Training: {i+1}/{len(self.train_indices)}", end="", flush=True)
                
                if i + 1 < len(self.train_indices):
                    future = self.executor.submit(self.train_reader.__getitem__, self.train_indices[i + 1])
                
                if i == 0:
                    batch = self.train_reader[train_ind].to(self.device)
                else:
                    batch = future.result().to(self.device)
                
                self.optimizer.zero_grad()
                
                _, loss = self.forward_step(batch)
                all_losses.append(loss.item())
                
                if self.use_wandb:
                    import wandb
                    wandb.log({'loss': loss.item()})
                
                loss.backward()
                
                if track_gradients:
                    gn = torch.sqrt(torch.stack([
                        p.grad.norm()**2 for p in self.model.parameters() if p.grad is not None
                    ]).sum())
                    gns.append(gn)
                
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            avg_loss = np.mean(all_losses) + 1e-8
            loss_record.append(avg_loss)
            epoch_record.append(ep)
            lr_record.append(self.scheduler.get_last_lr()[0])
            self.scheduler.step()
            
            self.model.eval()
            val_losses = []
            future_val = None
            
            for j, val_ind in enumerate(self.val_indices):
                print(f"\r\033[KEpoch {ep}/{epochs} - Training: {i+1}/{len(self.train_indices)} - Validation: {j+1}/{len(self.val_indices)}", end="", flush=True)
                
                if j + 1 < len(self.val_indices):
                    future_val = self.executor.submit(self.val_reader.__getitem__, self.val_indices[j + 1])
                
                if j == 0:
                    batch = self.val_reader[val_ind].to(self.device)
                else:
                    batch = future_val.result().to(self.device)
                
                with torch.no_grad():
                    _, loss = self.forward_step(batch)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            val_loss_record.append(val_loss)
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            minutes, seconds = divmod(int(epoch_duration), 60)
            time_str = f"{minutes:02}:{seconds:02}"
            time_record.append(time_str)
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': ep,
                    'ep_loss': avg_loss,
                    'ep_val_loss': val_loss,
                })
            
            print(" " * 120, end="\r")
            
            extra = ""
            if gns:
                extra = f"\t|  Grad: {sum(gns) / len(gns):.4f}"
            
            print(f'Loss: {round(avg_loss, 4)}\t|  Val Loss: {round(val_loss, 4)}\t|  LR: {round(self.scheduler.get_last_lr()[0], 7)}\t|  Time: {time_str}{extra}')
            
            if ep % checkpoint_every == 0:
                self._save_checkpoint(
                    trained_models_path, training_logs_path,
                    epoch_record, loss_record, val_loss_record, lr_record, time_record
                )
                print(f'✓ Checkpoint saved: {self.model_id}\n')
        
        self._save_checkpoint(
            trained_models_path, training_logs_path,
            epoch_record, loss_record, val_loss_record, lr_record, time_record
        )
        
        print("\n" + "=" * 60)
        print(f'✓ Training complete!')
        print(f'Model saved: {self.model_id}')
        print("=" * 60)
    
    def _save_checkpoint(self, trained_models_path, training_logs_path,
                         epoch_record, loss_record, val_loss_record, lr_record, time_record):
        """Save model checkpoint and logs."""
        loss_df = pd.DataFrame({
            'epoch': epoch_record,
            'loss': loss_record,
            'val_loss': val_loss_record,
            'lr': lr_record,
            'time': time_record
        })
        
        torch.save(self.model.state_dict(), trained_models_path / f"{self.model_id}.pth")
        loss_df.to_csv(training_logs_path / "loss_records" / f"{self.model_id}.csv")
        
        with open(training_logs_path / "configs" / f"{self.model_id}.json", 'w') as f:
            json.dump(self.network_dict, f, indent=4)
    
    def _parse_args(self, argv=None):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Train neural operator models'
        )
        parser.add_argument(
            'config', 
            metavar='<project>/<config_name>',
            help='Path to training config (e.g., ERA5/example_training)'
        )
        parser.add_argument('--wandb', '-wb', action='store_true',
                            help='Enable Weights & Biases logging')
        parser.add_argument('--gradient-clipping', dest='gradient_clipping', action='store_true',
                            help='Enable gradient clipping')
        parser.add_argument('--no-gradient-clipping', dest='gradient_clipping', action='store_false',
                            help='Disable gradient clipping')
        parser.add_argument('--track-gradients', dest='track_gradients', action='store_true',
                            help='Track and log gradient norms')
        parser.add_argument('--no-track-gradients', dest='track_gradients', action='store_false',
                            help='Disable gradient norm tracking')
        parser.add_argument('--checkpoint-every', type=int, default=None,
                            help='Save checkpoint every N epochs (default: 5)')
        parser.set_defaults(gradient_clipping=None, track_gradients=None)
        return parser.parse_args(argv)
