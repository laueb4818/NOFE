"""
Trainer package for Neural Operator Models.

Usage:
    from src.trainer import StandardTrainer, StandardTrainer_nolifting, SubgraphTrainer

    trainer = StandardTrainer()
    trainer.run(['ERA5/example_training'])

    trainer = StandardTrainer_nolifting()
    trainer.run(['ERA5/example_training'])

    trainer = SubgraphTrainer(use_gradient_clipping=True, track_gradients=True)
    trainer.run(['ERA5/example_subgraph_training', '--wandb'])
"""

from .base_trainer import BaseTrainer
from .standard_trainer import StandardTrainer
from .standard_trainer_nolifting import StandardTrainer_nolifting
from .subgraph_trainer import SubgraphTrainer

__all__ = ['BaseTrainer', 'StandardTrainer', 'StandardTrainer_nolifting', 'SubgraphTrainer']
