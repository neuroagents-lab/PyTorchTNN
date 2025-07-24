import torch
import torch.nn as nn

__all__ = ["HarborPolicyBase"]


class HarborPolicyBase(nn.Module):
    """
    Base class definition for all harbor policies. The desired output shape of the
    policy (`input_shape'), a list denoting the number of channels in each input
    (`in_channels'), and an int denoting the number of output channels (`out_channels')
    of this harbor policy are required.

    Args:
        input_shape (int, int): tells us the desired output spatial shape of this
            harbor policy. It is usually determined by the output spatial shape of
            the preceding recurrent module, as determined by the `shape_from' argument
            in the config file.
        in_channels [int]: list of ints telling us the number of input channels for
            each input to the harbor policy.
        out_channels (int): an int telling us the number of output channels of this
            harbor policy
    """

    def __init__(self, input_shape, in_channels, out_channels):
        super(HarborPolicyBase, self).__init__()

        assert input_shape is not None
        assert in_channels is not None
        assert out_channels is not None

        # support for habor policy between fc layers [len=0]
        assert len(input_shape) == 2 or len(input_shape) == 0, (
            "input_shape should be a tuple of length "
            + "2 for the spatial size of the harbor policy output."
        )

        self._input_shape = input_shape
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._operations_initialized = False
        self._input_shapes = None  # shapes of all inputs feeding into this policy

        # Assumes that each type of input uses the same operation type
        self.input_specific_ops = {
            "ff_input": None,
            "fb_input": None,
            "skip_input": None,
        }

    @property
    def input_shapes(self):
        assert self._input_shapes is not None
        return self._input_shapes

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def operations_initialized(self):
        return self._operations_initialized

    def random_input(self):
        # This property is only used during the initialization of the recurrent
        # module, which is why we can use the batch size of 10 shown here (batch
        # size is arbitrary). Using batch size of 10 instead of 1 so batch norm
        # does not raise an error.

        # support for harbor policy between fc layers [len=0]
        if len(self.input_shape) == 2:
            inp = torch.rand(
                10, self.out_channels, self.input_shape[0], self.input_shape[1]
            )
        elif len(self.input_shape) == 0:
            inp = torch.rand(10, self.out_channels)
        else:
            raise NotImplementedError(
                "len(self.input_shape) other than 2 or 0 is not allowed"
            )

        return inp

    def initialize_operations(self, input_shapes):
        """
        Args:
            input_shapes [(int, int, int), ...]: list of tuples where each tuple
                describes the shape of the input to the harbor policy in the format
                of (C, H, W).
        """
        raise NotImplementedError

    def spatial_op(self, inputs, input_types):
        raise NotImplementedError

    def channel_op(self, inputs, input_types):
        raise NotImplementedError

    def forward(self, inputs=None, input_types=None):
        # inputs [torch.Tensor]: the list of inputs to the recurrent module
        # input_types [str]: the list of input types for each input
        assert self.operations_initialized
        assert isinstance(inputs, list) and isinstance(input_types, list)
        assert len(inputs) == len(input_types)

        # There should only be one feedforward input
        assert len([x for x in input_types if x == "ff_input"]) == 1

        outputs = self.spatial_op(inputs, input_types)
        outputs = self.channel_op(outputs, input_types)

        return outputs
