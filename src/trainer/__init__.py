"""
Trainer package for Neural Operator Models.

Usage:
    from src.trainer import StandardTrainer, SubgraphTrainer
    
    trainer = StandardTrainer()
    trainer.run(['ERA5/example_training'])
    
    trainer = SubgraphTrainer(use_gradient_clipping=True, track_gradients=True)
    trainer.run(['ERA5/example_subgraph_training', '--wandb'])
"""

from .base_trainer import BaseTrainer
from .standard_trainer import StandardTrainer
from .subgraph_trainer import SubgraphTrainer
from .standard_trainer_nolifting import StandardTrainer_nolifting

__all__ = ['BaseTrainer', 'StandardTrainer', 'SubgraphTrainer']
