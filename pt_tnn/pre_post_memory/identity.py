from .memory_base import MemoryBase


__all__ = ["Identity"]


class Identity(MemoryBase):
    """
    Class to instantiate the "do nothing" operation.
    """

    def __init__(
        self, in_channels=None, out_channels=None, num_timesteps=None, **kwargs
    ):
        super(Identity, self).__init__(
            in_channels=in_channels, out_channels=in_channels
        )
        # since we are doing no transformation, the out_channels should equal the in_channels

    def forward(self, x, curr_timestep=None, **kwargs):
        return x
