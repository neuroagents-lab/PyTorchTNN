from .recurrent_cell_base import RecurrentCellBase
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ["UGRNNCell"]


class UGRNNCell(RecurrentCellBase):
    # from: https://github.com/neuroailab/convrnns/blob/master/convrnns/utils/cells.py#L723-L749

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
        layernorm=False,
        # parameters for RecurrentCellBase
        state_shape=None,
        # parameters for UGRNN
        forget_bias=1.0,
        **kwargs
    ):
        super(UGRNNCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=out_channels,
        )

        self.conv = nn.Conv2d(
            in_channels=input_in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
        )

        if init_dict is not None:
            init_method_name = init_dict.pop("method")
            init_method = getattr(init, init_method_name)
            init_method(self.conv.weight, **init_dict)
        if use_bias and bias is not None:
            init.constant_(self.conv.bias, val=bias)

        self.layernorm = layernorm
        if layernorm:
            self.ln_g_act = nn.LayerNorm(out_channels)
            self.ln_c_act = nn.LayerNorm(out_channels)

        self.forget_bias = forget_bias

    def forward(self, inputs, state, **kwargs):
        # should return output, state
        if state is None:
            state = torch.zeros_like(inputs, device=inputs.device)
            # we bypass the zero_state in RecurrentCellBase (which requires `state_shape` and `state_in_channels`
            # not None), because we assume the shapes of `inputs` and `state` are the same
        concat = self.conv(
            torch.concat([inputs, state], dim=1)
        )  # (bs, out_channels*2, H, W)
        g_act, c_act = torch.tensor_split(concat, 2, dim=1)
        # see: https://pytorch.org/docs/main/generated/torch.tensor_split.html#torch.tensor_split

        if self.layernorm:
            g_act = self.ln_g_act(g_act.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            c_act = self.ln_c_act(c_act.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        g = torch.sigmoid(g_act + self.forget_bias)
        c = torch.tanh(c_act)
        new_state = g * state + (1.0 - g) * c
        return new_state, new_state
