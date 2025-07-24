import torch.nn as nn

from .recurrent_cell_base import RecurrentCellBase

__all__ = ["ConvRNNBasicCell"]


class ConvRNNBasicCell(RecurrentCellBase):
    """
    Class definition of the basic convolutional RNN.

    Args:
        input_in_channels (int): number of input channels of the convolution operation
        out_channels (int): number of output channels of the convolution operation
        state_shape (int, int): spatial size of the state input
        activation (str): type of activation function. Must be defined in torch.nn.
            Default: "ReLU".
        ksize (int, int): kernel size of the convolution operation in the ConvRNN.
            Default: (3, 3).
        stride (int): stride of convolution. Default: 1.
        padding (int): padding of input to convolution. Default: 0.
        activation_func_kwargs (dict): keyword arguments for the activation function.
            Default: {}.
    """

    def __init__(
        self,
        input_in_channels=None,
        out_channels=None,
        state_shape=None,
        activation="ReLU",
        ksize=(3, 3),
        stride=1,
        padding=0,
        activation_func_kwargs={},
        **kwargs
    ):
        super(ConvRNNBasicCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=out_channels,
        )

        assert input_in_channels is not None
        assert out_channels is not None

        # Convolution operation for the input to the recurrent cell
        self.conv_input = nn.Conv2d(
            input_in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
        )

        # Convolution operation for the hidden state
        self.conv_state = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
        )

        # Activation function applied after the state and the input are combined
        self.activation = getattr(nn, activation)(**activation_func_kwargs)

    def forward(self, inputs, state, **kwargs):
        # Takes as input the inputs and the current state and outputs the recurrent
        # cell's output and new hidden state. The state's spatial size must be the
        # same as the inputs' spatial size.

        if state is None:
            state = self.zero_state(batch_size=inputs.shape[0], device=inputs.device)

        assert inputs.shape[0] == state.shape[0]
        assert inputs.shape[2:] == state.shape[2:]

        i = self.conv_input(inputs)
        s = self.conv_state(state)

        assert i.shape == s.shape
        new_state = self.activation(i + s)

        return new_state, new_state
