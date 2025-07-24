from .recurrent_cell_base import RecurrentCellBase
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ["IntersectionRNNCell"]


class IntersectionRNNCell(RecurrentCellBase):
    # from: https://github.com/neuroailab/convrnns/blob/b76a44d86ca9ab2c90821d0a5f281330188f699e/convrnns/utils/cells.py#L838-L883

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
        # parameters for IntersectionRNN
        forget_bias=1.0,
        **kwargs
    ):
        super(IntersectionRNNCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=out_channels,
        )

        self.conv = nn.Conv2d(
            in_channels=input_in_channels * 2,
            out_channels=out_channels * 4,
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
            self.ln_gh_act = nn.LayerNorm(out_channels)
            self.ln_h_act = nn.LayerNorm(out_channels)
            self.ln_gy_act = nn.LayerNorm(out_channels)
            self.ln_y_act = nn.LayerNorm(out_channels)

        self.forget_bias = forget_bias

    def forward(self, inputs, state, **kwargs):
        # should return output, state
        if state is None:
            state = torch.zeros_like(inputs, device=inputs.device)
            # we bypass the zero_state in RecurrentCellBase (which requires `state_shape` and `state_in_channels`
            # not None), because we assume the shapes of `inputs` and `state` are the same
        concat = self.conv(
            torch.concat([inputs, state], dim=1)
        )  # (bs, out_channels*4, H, W)
        gh_act, h_act, gy_act, y_act = torch.tensor_split(concat, 4, dim=1)
        # see: https://pytorch.org/docs/main/generated/torch.tensor_split.html#torch.tensor_split

        if self.layernorm:
            gh_act = self.ln_gh_act(gh_act.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            h_act = self.ln_h_act(h_act.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            gy_act = self.ln_gy_act(gy_act.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            y_act = self.ln_y_act(y_act.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        gh = torch.sigmoid(gh_act + self.forget_bias)
        h = torch.tanh(h_act)
        y = torch.relu(y_act)
        gy = torch.sigmoid(gy_act + self.forget_bias)

        new_y = gy * inputs + (1.0 - gy) * y  # passed through depth
        new_state = gh * state + (1.0 - gh) * h  # passed through time

        return new_y, new_state
