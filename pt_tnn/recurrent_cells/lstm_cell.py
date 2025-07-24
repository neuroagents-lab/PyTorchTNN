from .recurrent_cell_base import RecurrentCellBase
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ["LSTMCell"]


class LSTMCell(RecurrentCellBase):
    # from: https://github.com/neuroailab/convrnns/blob/b76a44d86ca9ab2c90821d0a5f281330188f699e/convrnns/utils/cells.py#L450

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
        # parameters for LSTM
        forget_bias=1.0,
        **kwargs
    ):
        super(LSTMCell, self).__init__(
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
            self.ln_input = nn.LayerNorm(out_channels)
            self.ln_transform = nn.LayerNorm(out_channels)
            self.ln_forget = nn.LayerNorm(out_channels)
            self.ln_output = nn.LayerNorm(out_channels)
            self.ln_state = nn.LayerNorm(out_channels)

        self.forget_bias = forget_bias

        self.activation = (
            getattr(nn, activation)(**activation_func_kwargs)
            if activation is not None
            else nn.Identity()
        )

    def forward(
        self, inputs, state, **kwargs
    ):  # currently, we are not supporting peepholes
        # should return output, state
        if state is None:
            state = torch.zeros_like(
                torch.concat([inputs, inputs], dim=1), device=inputs.device
            )
            # we bypass the zero_state in RecurrentCellBase (which requires `state_shape` and `state_in_channels`
            # not None), because we assume the shapes of `inputs` and `state` are the same

        c, h = torch.tensor_split(state, 2, dim=1)

        concat = self.conv(
            torch.concat([inputs, h], dim=1)
        )  # (bs, out_channels*4, H, W)
        i, j, f, o = torch.tensor_split(concat, 4, dim=1)
        # see: https://pytorch.org/docs/main/generated/torch.tensor_split.html#torch.tensor_split

        if self.layernorm:
            i = self.ln_input(i.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            j = self.ln_transform(j.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            f = self.ln_forget(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            o = self.ln_output(o.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(
            i
        ) * self.activation(j)

        if self.layernorm:
            new_c = self.ln_state(new_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        new_h = self.activation(new_c) * torch.sigmoid(o)
        new_state = torch.concat([new_c, new_h], dim=1)

        return new_h, new_state
