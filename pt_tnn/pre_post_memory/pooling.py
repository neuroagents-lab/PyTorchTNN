import torch
import torch.nn as nn
from .memory_base import MemoryBase

__all__ = ["MaxPool", "AdaptiveAvgPool"]


class MaxPool(MemoryBase):
    """
    Performs the max pool operation

    Args:
        ksize (int, int): size of the kernel over which to perform max pool.
            Default: (3, 3).
        stride (int): stride size for max pool operation. Default: 1.
        padding (int): padding for max pool operation. Default: 1.
    """

    def __init__(
        self,
        ksize,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        **kwargs
    ):
        super(MaxPool, self).__init__(in_channels=None, out_channels=None)

        self.maxpool = nn.MaxPool2d(
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x, curr_timestep=None, **kwargs):
        return self.maxpool(x)


class AdaptiveAvgPool(MemoryBase):
    """
    Performs the adaptive average pooling operation

    Args:
        output_size (int, int): desired spatial size of the output of this operation.
            Default: (6, 6).
    """

    def __init__(self, output_size=(6, 6), flatten=False, **kwargs):
        super(AdaptiveAvgPool, self).__init__(in_channels=None, out_channels=None)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        self.flatten = flatten

    def forward(self, x, curr_timestep=None, **kwargs):
        output = self.avgpool(x)
        if self.flatten:
            output = torch.flatten(output, start_dim=1)
        return output
