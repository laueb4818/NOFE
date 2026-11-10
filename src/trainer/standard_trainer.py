"""
Standard Trainer for KernelNN Models.

Trains standard graph neural network models on weather/satellite data.
"""

from .base_trainer import BaseTrainer
from src.utilities import (
    KernelNN,
    get_output_affinities,
    custom_KL_loss,
)


class StandardTrainer(BaseTrainer):
    """Trainer for standard KernelNN models."""
    
    def __init__(
        self,
        use_gradient_clipping=None,
        track_gradients=None,
        checkpoint_every=None,
    ):
        """
        Initialize standard trainer.
        
        Args:
            use_gradient_clipping: Apply gradient clipping (default: False)
            track_gradients: Track and log gradient norms (default: False)
            checkpoint_every: Save checkpoint every N epochs (default: 5)
        """
        super().__init__(
            use_gradient_clipping=use_gradient_clipping,
            track_gradients=track_gradients,
            checkpoint_every=checkpoint_every,
        )
    
    def create_model(self, model_dict):
        """Create KernelNN model."""
        return KernelNN(
            width=model_dict['width'],
            ker_width=model_dict['ker_width'],
            depth=model_dict['depth'],
            ker_in=model_dict['edge_attr'],
            in_width=model_dict['node_features'],
            out_width=model_dict['out_width'],
        )
    
    def forward_step(self, batch):
        """Forward pass for standard model."""
        out = self.model(batch)
        edge_index = batch.edge_index
        q = get_output_affinities(out, edge_index)
        loss = custom_KL_loss(batch.input_affinities, q).mean()
        return out, loss
