from .memory_base import MemoryBase
import torch.nn as nn

__all__ = ["ActivationLayer"]


class ActivationLayer(MemoryBase):
    """
    Class to instantiate the activation operation.
    """

    def __init__(self, activation="ReLU", activation_func_kwargs={}, **kwargs):
        super(ActivationLayer, self).__init__(in_channels=None, out_channels=None)
        self.activation = getattr(nn, activation)(**activation_func_kwargs)

    def forward(self, x, curr_timestep=None, **kwargs):
        return self.activation(x)
