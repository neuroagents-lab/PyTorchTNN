import torch.nn as nn
from typing import Any

__all__ = ["MemoryBase"]


class MemoryBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MemoryBase, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        # for modules in pre_memory, in_channels & out_channels can't be None
        # for modules in post_memory, in_channels & out_channels have no restrictions

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, *inputs: Any):
        raise NotImplementedError
