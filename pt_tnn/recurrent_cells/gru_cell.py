from .recurrent_cell_base import RecurrentCellBase
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ["GRUCell"]


class GRUCell(RecurrentCellBase):
    # from: https://github.com/neuroailab/convrnns/blob/b76a44d86ca9ab2c90821d0a5f281330188f699e/convrnns/utils/cells.py#L330

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
        # parameters for activation
        activation="Tanh",
        activation_func_kwargs={},
        # parameters for GRU
        forget_bias=1.0,
        **kwargs
    ):
        super(GRUCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=out_channels,
        )

        self.conv_ru = nn.Conv2d(
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

        self.conv_c = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
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
            init_method(self.conv_ru.weight, **init_dict)
            init_method(self.conv_c.weight, **init_dict)

        if use_bias and bias is not None:
            init.constant_(self.conv_ru.bias, val=bias)
            init.constant_(self.conv_c.bias, val=bias)

        self.layernorm = layernorm
        if layernorm:
            self.ln_r_pre = nn.LayerNorm(out_channels)
            self.ln_u_pre = nn.LayerNorm(out_channels)
            self.ln_c_pre = nn.LayerNorm(out_channels)

        self.forget_bias = forget_bias

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
        concat = self.conv_ru(
            torch.concat([inputs, state], dim=1)
        )  # (bs, out_channels*2, H, W)
        r_pre, u_pre = torch.tensor_split(concat, 2, dim=1)
        # see: https://pytorch.org/docs/main/generated/torch.tensor_split.html#torch.tensor_split

        if self.layernorm:
            r_pre = self.ln_r_pre(r_pre.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            u_pre = self.ln_u_pre(u_pre.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        r = torch.sigmoid(r_pre + self.forget_bias)
        u = torch.sigmoid(u_pre)

        c_pre = self.conv_c(
            torch.concat([inputs, r * state], dim=1)
        )  # (bs, out_channels, H, W)
        if self.layernorm:
            c_pre = self.ln_c_pre(c_pre.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        c = self.activation(c_pre)

        new_state = u * state + (1 - u) * c
        return new_state, new_state
