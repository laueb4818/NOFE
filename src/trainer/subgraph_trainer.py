"""
Subgraph Trainer for KernelNN_subgraph Models.

Trains subgraph-based graph neural network models where input and output
graphs have different node sets.
"""

from .base_trainer import BaseTrainer
from src.utilities import (
    KernelNN_subgraph,
    get_output_affinities,
    custom_KL_loss,
)


class SubgraphTrainer(BaseTrainer):
    """Trainer for KernelNN_subgraph models."""
    
    def __init__(
        self,
        use_gradient_clipping=None,
        track_gradients=None,
        checkpoint_every=None,
    ):
        """
        Initialize subgraph trainer.
        
        Args:
            use_gradient_clipping: Apply gradient clipping (default: True)
            track_gradients: Track and log gradient norms (default: False)
            checkpoint_every: Save checkpoint every N epochs (default: 5)
        """
        super().__init__(
            use_gradient_clipping=use_gradient_clipping,
            track_gradients=track_gradients,
            checkpoint_every=checkpoint_every,
        )
    
    def create_model(self, model_dict):
        """Create KernelNN_subgraph model."""
        input_node_dim = model_dict.get('input_node_features', model_dict.get('node_features', 1))
        output_node_dim = model_dict.get('output_node_features', model_dict.get('node_features', 1))
        
        return KernelNN_subgraph(
            width=model_dict['width'],
            ker_width=model_dict['ker_width'],
            depth=model_dict['depth'],
            ker_in=model_dict['edge_attr'],
            known_node_dim=input_node_dim,
            unknown_node_dim=output_node_dim,
            out_width=model_dict['out_width'],
        )
    
    def forward_step(self, batch):
        """Forward pass for subgraph model using target graph edges."""
        out = self.model(batch)
        edge_index = batch.target.edge_index
        q = get_output_affinities(out, edge_index)
        loss = custom_KL_loss(batch.target.input_affinities, q).mean()
        return out, loss
