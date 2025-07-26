from .recurrent_cell_base import RecurrentCellBase
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ["VanillaRNNCell"]


class VanillaRNNCell(RecurrentCellBase):
    # from: https://github.com/neuroailab/convrnns/blob/master/convrnns/utils/cells.py#L248-L327

    def __init__(
            self,
            # Conv2d arguments
            input_in_channels,
            out_channels,
            ksize,
            stride=1,
            padding="same",
            # following: https://github.com/neuroailab/convrnns/blob/master/convrnns/utils/cell_utils.py#L59
            dilation=1,
            groups=1,
            # parameters for bias_init
            use_bias=True,
            bias=None,
            padding_mode="zeros",
            # parameters for kernel_init
            init_dict=None,
            # parameters for layer-norm
            layernorm=True,
            # parameters for RecurrentCellBase
            state_shape=None,
            # parameters for activation
            activation="ELU",
            activation_func_kwargs={},
            **kwargs
    ):
        super(VanillaRNNCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=out_channels,
        )

        self.conv_input = nn.Conv2d(
            in_channels=input_in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
        )

        self.conv_state = nn.Conv2d(
            in_channels=input_in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
        )
        # we assume input and state channels are the same

        if init_dict is not None:
            init_method_name = init_dict.pop("method")
            init_method = getattr(init, init_method_name)
            init_method(self.conv_input.weight, **init_dict)
            init_method(self.conv_state.weight, **init_dict)

        if use_bias and bias is not None:
            init.constant_(self.conv_input.bias, val=bias)
            init.constant_(self.conv_state.bias, val=bias)

        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm(out_channels)

        self.activation = (
            getattr(nn, activation)(**activation_func_kwargs)
            if activation is not None
            else nn.Identity()
        )

    def forward(self, inputs, state, **kwargs):
        # should return output, state
        if state is None:
            state = torch.zeros_like(inputs, device=inputs.device)
            # we bypass the zero_state in RecurrentCellBase (which requires `state_shape` and `state_in_channels`
            # not None), because we assume the shapes of `inputs` and `state` are the same

        i = self.conv_input(inputs)
        s = self.conv_state(state)

        new_state = i + s

        if self.layernorm:
            new_state = self.ln(new_state.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        new_state = self.activation(new_state)

        return new_state, new_state
