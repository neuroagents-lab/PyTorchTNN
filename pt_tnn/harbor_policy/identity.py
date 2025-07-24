import torch
import numpy as np

from .harbor_policy_base import HarborPolicyBase

__all__ = ["Identity"]


class Identity(HarborPolicyBase):
    """
    Harbor policy that performs a concatenation over all the inputs without applying
    any operation on the inputs. Thus, this module assumes that the spatial size of
    all the inputs are the same.

    Args:
        input_shape (int, int): a tuple of ints that determines the exact spatial
            size to which to resize inputs, as determined by the shape of the output
            of the prior feedforward recurrent module. Default: None.
        in_channels [int]: a list of total number of channels for each input to this
            module. Default: None.
    """

    def __init__(self, input_shape=None, in_channels=None):
        super(Identity, self).__init__(
            in_channels=in_channels,
            input_shape=input_shape,
            out_channels=int(np.sum(in_channels)),
        )

    def initialize_operations(self, input_shapes):
        self._input_shapes = input_shapes
        self._operations_initialized = True

    def spatial_op(self, inputs, input_types):
        # Check spatial sizes
        shape = inputs[0].shape[2:]
        for inp in inputs:
            assert (
                inp.ndim == 4 or inp.ndim == 2
            )  # (N, C, H, W) or (N, C) [fully-connected]
            assert (
                inp.shape[2:] == shape
            ), f"Unexpected spatial size of {inp.shape[2:]}, expected {shape}"

        return inputs

    def channel_op(self, inputs, input_types):
        # Concatenate along channel dimension
        inputs = torch.cat(inputs, dim=1)

        return inputs
