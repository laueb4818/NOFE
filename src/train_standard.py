#!/usr/bin/env python3
"""
Training script for Standard KernelNN models on regular datasets.

Usage:
    python src/train_standard.py <project>/<config_name> [options]

The positional argument resolves to <project>/training_configs/<config_name>.json.

Examples:
    python src/train_standard.py ERA5/regular_papermodel1
    python src/train_standard.py ERA5/regular_papermodel1 --wandb
    python src/train_standard.py ERA5/regular_papermodel1 --gradient-clipping --checkpoint-every 10

Options:
    <project>/<config_name>               Training config, resolved as above
    --wandb / -wb                         Enable Weights & Biases logging
    --gradient-clipping / --no-gradient-clipping
    --track-gradients / --no-track-gradients
    --checkpoint-every <N>                Save checkpoint every N epochs (default: 5)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.trainer import StandardTrainer


def main():
    trainer = StandardTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
