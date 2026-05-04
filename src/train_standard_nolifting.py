#!/usr/bin/env python3
"""
Training script for Standard KernelNN models.

Usage:
    python src/train_standard.py --config <project>/<config_name> [options]

Examples:
    python src/train_standard.py --config ERA5/example_training
    python src/train_standard.py --config ERA5/example_training --wandb
    python src/train_standard.py --config ERA5/example_training --gradient-clipping --checkpoint-every 10

Options:
    --config <project>/<config_name>    Path to training config
    --wandb                               Enable Weights & Biases logging
    --gradient-clipping / --no-gradient-clipping
    --track-gradients / --no-track-gradients
    --checkpoint-every <N>               Save checkpoint every N epochs (default: 5)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.trainer import StandardTrainer_nolifting


def main():
    trainer = StandardTrainer_nolifting()
    trainer.run()


if __name__ == "__main__":
    main()
