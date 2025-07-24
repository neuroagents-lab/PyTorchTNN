import torch
import numpy as np

from torchvision.transforms import Resize, InterpolationMode

from .harbor_policy_base import HarborPolicyBase

__all__ = ["ResizeConcat"]


class ResizeConcat(HarborPolicyBase):
    """
    Harbor policy that resizes the inputs in the spatial dimension and then
    concatenates them all along the channel dimension. Note that the same
    resize operation is performed on all input types, no matter the type of
    input.

    Args:
        input_shape (int, int): a tuple of ints that determines the exact spatial
            size to which to resize inputs, as determined by the shape of the output
            of the prior feedforward recurrent module. Default: None.
        in_channels [int]: a list of total number of channels for each input to this
            module. Default: None.
        interpolation (InterpolationMode): how to perform interpolation for the resize
            operation. Default: InterpolationMode.BILINEAR.
    """

    def __init__(
        self,
        input_shape=None,
        in_channels=None,
        interpolation=InterpolationMode.BILINEAR,
    ):
        super(ResizeConcat, self).__init__(
            in_channels=in_channels,
            input_shape=input_shape,
            out_channels=int(np.sum(in_channels)),
        )

        self._interpolation = interpolation

    def initialize_operations(self, input_shapes):
        self._input_shapes = input_shapes

        # Define the resize operation
        resize_op = Resize(self.input_shape, interpolation=self._interpolation)

        # Resize operation is the same for all input types in this particular
        # harbor policy.
        for input_type in self.input_specific_ops:
            self.input_specific_ops[input_type] = resize_op

        self._operations_initialized = True

    def spatial_op(self, inputs, input_types):
        # Resize each input
        for i, (inp, inp_type) in enumerate(zip(inputs, input_types)):
            # Updating inputs in-place
            assert inp_type in self.input_specific_ops.keys()
            curr_op = self.input_specific_ops[inp_type]
            assert curr_op is not None
            inputs[i] = curr_op(inp)

        return inputs

    def channel_op(self, inputs, input_types):
        # Concatenate along channel dimension
        inputs = torch.cat(inputs, dim=1)

        return inputs
