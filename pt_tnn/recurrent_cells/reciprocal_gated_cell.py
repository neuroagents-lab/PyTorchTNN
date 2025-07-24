from typing import Union
from .recurrent_cell_base import RecurrentCellBase
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ["ReciprocalGatedCell"]


def init_activation(activation, activation_func_kwargs):
    return (
        getattr(nn, activation)(**activation_func_kwargs)
        if activation is not None
        else nn.Identity()
    )


class CReLU(nn.Module):
    # taken from https://gist.github.com/lintangsutawika/f2f3fb422d6d7df28bd74e26940da2e6,
    # tf implementation for verification:
    # https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/nn_ops.py#L3592

    def __init__(self, dim=-1):
        super(CReLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.cat([x, -x], dim=self.dim)
        return F.relu(x)


class Conv2dOp(nn.Module):
    """
    Defines the normal 2d convolution operation.

    Args:
        in_channels (int): number of input channels of the convolution operation
        out_channels (int): number of output channels of the convolution operation
        ksize (int, int): kernel size of the convolution operation. Default: (3, 3).
        stride (int): stride of the convolution kernel. Default: [1, 1, 1, 1].
        padding (int): how much to pad the input on all sides. Default: 1.
        dropout (float): probability of dropout. Default: 0.
        batchnorm (int): number of features for the batch norm operation. Default: None.
        batchnorm_timevary (bool): whether to vary the batch norm operation across time.
            Default: False.
        num_timesteps (int or None): number of timesteps to unroll the temporal graph.
            Default: None.
    """

    def __init__(
        self,
        # Conv2d arguments
        in_channels,
        out_channels,
        ksize,
        stride=1,
        padding: Union[str, int] = 0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        # parameters for bias_init
        use_bias=True,
        bias=None,
        # parameters for kernel_init
        init_dict=None,
        # dropout, batchnorm, activation arguments
        dropout=0.0,
        batchnorm=None,
        batchnorm_timevary=False,
        # parameters for activation
        activation=None,
        activation_func_kwargs={},
        num_timesteps=None,
        **kwargs,
    ):
        super(Conv2dOp, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
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
            init_method(self.conv.weight, **init_dict)
        if use_bias and bias is not None:
            init.constant_(self.conv.bias, val=bias)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.num_timesteps = num_timesteps
        self.batchnorm_timevary = batchnorm_timevary
        if batchnorm_timevary:
            print(f"using time-dependent BatchNorm for {num_timesteps} timesteps...")
            assert num_timesteps is not None
            self.bns = nn.ModuleList(
                [
                    (
                        nn.BatchNorm2d(batchnorm)
                        if batchnorm is not None
                        else nn.Identity()
                    )
                    for _ in range(num_timesteps)
                ]
            )
        else:
            self.bn = (
                nn.BatchNorm2d(batchnorm) if batchnorm is not None else nn.Identity()
            )

        self.activation = (
            getattr(nn, activation)(**activation_func_kwargs)
            if activation is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, curr_timestep=None, **kwargs):
        """
        Args:
            x (torch.Tensor): the inputs to the pre-memory operations. This is
                the output of the harbor policy, which combines the inputs to
                the recurrent module.
            curr_timestep (int): current timestep
        """
        if not self.batchnorm_timevary:
            curr_bn = self.bn
        else:
            assert curr_timestep is not None
            curr_bn = self.bns[curr_timestep]
        output = curr_bn(self.conv(self.dropout(x)))

        return self.activation(output)


class DSConv2dOp(nn.Module):
    """
    Defines the 2d depth separate convolution operation.

    Args:
        in_channels (int): number of input channels of the convolution operation
        out_channels (int): number of output channels of the convolution operation
        ksize (int, int): kernel size of the convolution operation. Default: (3, 3).
        stride (int): stride of the convolution kernel. Default: [1, 1, 1, 1].
        padding (int): how much to pad the input on all sides. Default: 1.
        dropout (float): probability of dropout. Default: 0.
        batchnorm (int): number of features for the batch norm operation. Default: None.
        batchnorm_timevary (bool): whether to vary the batch norm operation across time.
            Default: False.
        num_timesteps (int or None): number of timesteps to unroll the temporal graph.
            Default: None.
    """

    def __init__(
        self,
        # Conv2d arguments
        in_channels,
        out_channels,
        ksize,
        stride=1,
        padding: Union[str, int] = 0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        # other parameters
        ch_mult=1,
        repeat=False,
        intermediate_activation="ELU",
        intermediate_activation_func_kwargs={},
        # parameters for bias_init
        use_bias=True,
        bias=None,
        # parameters for kernel_init
        init_dict=None,
        # dropout, batchnorm, activation arguments
        dropout=0.0,
        batchnorm=None,
        batchnorm_timevary=False,
        # parameters for activation
        activation=None,
        activation_func_kwargs={},
        num_timesteps=None,
        **kwargs,
    ):
        super(DSConv2dOp, self).__init__()
        # see: https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d
        self.depth_wise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * ch_mult,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=use_bias,
            padding_mode=padding_mode,
        )
        self.point_wise = nn.Conv2d(
            in_channels=in_channels * ch_mult,
            out_channels=out_channels,
            kernel_size=1,
            padding="valid"
            # https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/nn_impl.py#L996
        )

        self.repeat = repeat
        if repeat:
            self.repeat_depth = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * ch_mult,
                kernel_size=ksize,
                stride=1,
                padding="same",
                dilation=dilation,
                groups=in_channels,
                bias=use_bias,
                padding_mode=padding_mode,
            )
            self.inter_act = init_activation(
                activation=intermediate_activation,
                activation_func_kwargs=intermediate_activation_func_kwargs,
            )

        if init_dict is not None:
            init_method_name = init_dict.pop("method")
            init_method = getattr(init, init_method_name)
            init_method(self.depth_wise.weight, **init_dict)
            init_method(self.point_wise.weight, **init_dict)
            if repeat:
                init_method(self.repeat_depth.weight, **init_dict)

        if use_bias and bias is not None:
            init.constant_(self.depth_wise.bias, val=bias)
            init.constant_(self.point_wise.bias, val=bias)
            if repeat:
                init.constant_(self.repeat_depth.bias, val=bias)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.num_timesteps = num_timesteps
        self.batchnorm_timevary = batchnorm_timevary
        if batchnorm_timevary:
            print(f"using time-dependent BatchNorm for {num_timesteps} timesteps...")
            assert num_timesteps is not None
            self.bns = nn.ModuleList(
                [
                    (
                        nn.BatchNorm2d(batchnorm)
                        if batchnorm is not None
                        else nn.Identity()
                    )
                    for _ in range(num_timesteps)
                ]
            )
        else:
            self.bn = (
                nn.BatchNorm2d(batchnorm) if batchnorm is not None else nn.Identity()
            )

        self.activation = (
            getattr(nn, activation)(**activation_func_kwargs)
            if activation is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, curr_timestep=None, **kwargs):
        """
        Args:
            x (torch.Tensor): the inputs to the pre-memory operations. This is
                the output of the harbor policy, which combines the inputs to
                the recurrent module.
            curr_timestep (int): current timestep
        """
        if not self.batchnorm_timevary:
            curr_bn = self.bn
        else:
            assert curr_timestep is not None
            curr_bn = self.bns[curr_timestep]

        if self.repeat:
            x = self.repeat_depth(x)  # extra depth-wise conv
            x = self.inter_act(x)  # activation

        x = self.depth_wise(self.dropout(x))
        x = self.point_wise(x)

        output = curr_bn(x)

        return self.activation(output)


class ReciprocalGatedCell(RecurrentCellBase):
    # https://github.com/neuroailab/convrnns/blob/b76a44d86ca9ab2c90821d0a5f281330188f699e/convrnns/models/rgc_shallow.json
    """
    memory with cell and output that both, by default, perfectly integrate incoming information;
    the cell then gates the input to the output, and the output gates the input to the cell.
    both cell and output are pseudo-residual recurrent networks, as well.
    """

    def __init__(
        self,
        # Conv2d arguments
        input_in_channels,
        out_channels,
        cell_depth,
        res_channels,
        fb_channels=None,
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
        # dropout, batchnorm, activation arguments
        dropout=0.0,
        batchnorm=None,
        batchnorm_timevary=False,
        num_timesteps=None,
        # parameters for layer-norm
        layernorm=False,
        # parameters for RecurrentCellBase
        state_shape=None,
        # parameters for ReciprocalGateRNN
        tau_filter_size=[3, 3],
        gate_filter_size=[3, 3],
        ff_filter_size=[3, 3],
        in_out_filter_size=[3, 3],
        cell_tau_filter_size=None,
        feedback_filter_size=[3, 3],  # [1, 1] in median_rgc but [3, 3] in shallow_rgc
        feedback_entry="out",
        feedback_depth_separable=False,  # True in median_rgc but not shallow_rgc
        ff_depth_separable=False,
        in_out_depth_separable=False,
        gate_depth_separable=False,  # True in median_rgc but not shallow_rgc
        tau_depth_separable=False,  # True in median_rgc but not shallow_rgc
        tau_nonlinearity="Sigmoid",
        tau_nonlinearity_kwargs={},
        gate_nonlinearity="Sigmoid",  # tf.tanh in median_rgc but not shallow_rgc
        gate_nonlinearity_kwargs={},
        tau_bias=0.0,  # different from 0.0 in median_rgc but not shallow_rgc
        gate_bias=0.0,  # different from 0.0 in median_rgc but not shallow_rgc
        tau_multiplier=-1.0,
        gate_multiplier=-1.0,
        tau_offset=1.0,
        gate_offset=1.0,
        input_activation="Identity",  # tf.nn.elu in median_rgc but not shallow_rgc
        input_activation_kwargs={},
        feedback_activation="Identity",
        feedback_activation_kwargs={},
        cell_activation="ELU",
        cell_activation_kwargs={},
        out_activation="ELU",
        out_activation_kwargs={},
        cell_residual=False,
        out_residual=False,  # True in median_rgc but not shallow_rgc
        residual_to_cell_tau=False,
        residual_to_cell_gate=False,
        residual_to_out_tau=False,
        residual_to_out_gate=True,  # False in median_rgc but not shallow_rgc
        input_to_tau=False,
        input_to_gate=False,
        input_to_cell=True,
        input_to_out=False,  # True in median_rgc but not shallow_rgc
        cell_to_out=False,
        ds_repeat=False,
    ):
        """
        Initialize the memory function of the ReciprocalGateCell.
        """
        super(ReciprocalGatedCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=out_channels,
        )

        self.tau_filter_size = tau_filter_size
        self.gate_filter_size = gate_filter_size
        self.cell_tau_filter_size = (
            cell_tau_filter_size
            if cell_tau_filter_size is not None
            else tau_filter_size
        )
        self.ff_filter_size = ff_filter_size
        self.feedback_filter_size = feedback_filter_size
        self.in_out_filter_size = in_out_filter_size

        self.tau_depth_separable = tau_depth_separable
        self.ff_depth_separable = ff_depth_separable
        self.gate_depth_separable = gate_depth_separable
        self.in_out_depth_separable = in_out_depth_separable

        if self.gate_filter_size == [0, 0]:
            self.use_cell = False
        else:
            self.use_cell = True

        self.feedback_entry = feedback_entry
        self.feedback_depth_separable = feedback_depth_separable
        self.cell_depth = cell_depth
        self.out_depth = out_channels
        self.cell_residual = cell_residual
        self.out_residual = out_residual
        self.residual_to_cell_tau = residual_to_cell_tau
        self.residual_to_cell_gate = residual_to_cell_gate
        self.residual_to_out_tau = residual_to_out_tau
        self.residual_to_out_gate = residual_to_out_gate
        self.input_to_tau = input_to_tau
        self.input_to_gate = input_to_gate
        self.input_to_cell = input_to_cell
        self.input_to_out = input_to_out
        self.cell_to_out = cell_to_out
        self.ds_repeat = ds_repeat

        self._tau_bias = tau_bias
        self._gate_bias = gate_bias
        self._tau_offset = tau_offset
        self._gate_offset = gate_offset
        self._tau_k = tau_multiplier
        self._gate_k = gate_multiplier

        self._input_activation = init_activation(
            activation=input_activation, activation_func_kwargs=input_activation_kwargs
        )

        if cell_activation == "CReLU":
            print("using CReLU! doubling cell out depth")
            self.cell_depth_out = 2 * self.cell_depth
            self._cell_activation = CReLU()
        else:
            self.cell_depth_out = self.cell_depth
            self._cell_activation = init_activation(
                activation=cell_activation,
                activation_func_kwargs=cell_activation_kwargs,
            )

        self._gate_nonlinearity = init_activation(
            activation=gate_nonlinearity,
            activation_func_kwargs=gate_nonlinearity_kwargs,
        )
        self._tau_nonlinearity = init_activation(
            activation=tau_nonlinearity, activation_func_kwargs=tau_nonlinearity_kwargs
        )

        self._out_activation = init_activation(
            activation=out_activation, activation_func_kwargs=out_activation_kwargs
        )

        def creat_conv_op(
            in_dim,
            out_dim,
            ksize,
            activation,
            activation_func_kwargs,
            depth_sep,
            dropout,
        ):
            if depth_sep:
                conv_op = DSConv2dOp(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    ksize=ksize,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding_mode,
                    ch_mult=1,
                    repeat=ds_repeat,
                    intermediate_activation="ELU",
                    intermediate_activation_func_kwargs={},
                    use_bias=use_bias,
                    bias=bias,
                    init_dict=init_dict,
                    dropout=dropout,
                    batchnorm=batchnorm,
                    batchnorm_timevary=batchnorm_timevary,
                    num_timesteps=num_timesteps,
                    activation=activation,
                    activation_func_kwargs=activation_func_kwargs,
                )
            else:
                conv_op = Conv2dOp(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    ksize=ksize,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding_mode,
                    use_bias=use_bias,
                    bias=bias,
                    init_dict=init_dict,
                    dropout=dropout,
                    batchnorm=batchnorm,
                    batchnorm_timevary=batchnorm_timevary,
                    num_timesteps=num_timesteps,
                    activation=activation,
                    activation_func_kwargs=activation_func_kwargs,
                )
            return conv_op

        if fb_channels is not None:
            self.feedback_conv = creat_conv_op(
                in_dim=fb_channels,
                out_dim=input_in_channels,
                ksize=self.feedback_filter_size,
                activation=feedback_activation,
                activation_func_kwargs=feedback_activation_kwargs,
                depth_sep=self.feedback_depth_separable,
                dropout=dropout,
            )

        if self.input_to_out:
            self.inp2out_conv = creat_conv_op(
                in_dim=input_in_channels,
                out_dim=self.out_depth,
                ksize=self.in_out_filter_size,
                activation=None,
                activation_func_kwargs={},
                depth_sep=self.in_out_depth_separable,
                dropout=0.0,
            )

        self.out_tau_conv = creat_conv_op(
            in_dim=self.out_depth,
            out_dim=self.out_depth,
            ksize=self.tau_filter_size,
            activation=None,
            activation_func_kwargs={},
            depth_sep=self.tau_depth_separable,
            dropout=dropout,
        )

        if self.residual_to_out_gate:
            self.out_gate_conv = creat_conv_op(
                in_dim=res_channels,
                out_dim=self.out_depth,
                ksize=self.gate_filter_size,
                activation=None,
                activation_func_kwargs={},
                depth_sep=self.gate_depth_separable,
                dropout=dropout,
            )

        if self.use_cell:
            if self.cell_residual:
                self.res2cell_conv = creat_conv_op(
                    in_dim=res_channels,
                    out_dim=self.cell_depth,
                    ksize=self.ff_filter_size,
                    activation=None,
                    activation_func_kwargs={},
                    depth_sep=self.ff_depth_separable,
                    dropout=dropout,
                )

            if self.input_to_cell:
                self.inp2cell_conv = creat_conv_op(
                    in_dim=input_in_channels,
                    out_dim=self.cell_depth,
                    ksize=self.ff_filter_size,
                    activation=None,
                    activation_func_kwargs={},
                    depth_sep=self.ff_depth_separable,
                    dropout=dropout,
                )

            self.cell_tau_conv = creat_conv_op(
                in_dim=self.cell_depth,
                out_dim=self.cell_depth,
                ksize=self.cell_tau_filter_size,
                activation=tau_nonlinearity,
                activation_func_kwargs=tau_nonlinearity_kwargs,
                depth_sep=self.tau_depth_separable,
                dropout=dropout,
            )

            self.cell_gate_conv = creat_conv_op(
                in_dim=self.out_depth,
                out_dim=self.cell_depth,
                ksize=self.gate_filter_size,
                activation=gate_nonlinearity,
                activation_func_kwargs=gate_nonlinearity_kwargs,
                depth_sep=self.gate_depth_separable,
                dropout=dropout,
            )

            self.cell2out_conv = creat_conv_op(
                in_dim=self.cell_depth,
                out_dim=self.out_depth,
                ksize=self.gate_filter_size,
                activation=None,
                activation_func_kwargs={},
                depth_sep=self.gate_depth_separable,
                dropout=dropout,
            )

        if self.out_residual:
            self.res2out_conv = creat_conv_op(
                in_dim=res_channels,
                out_dim=self.out_depth,
                ksize=[1, 1],
                activation=None,
                activation_func_kwargs={},
                depth_sep=False,
                dropout=dropout,
            )

        self.layernorm = layernorm
        if layernorm:
            self.ln_cell = nn.LayerNorm(self.cell_depth)
            self.ln_out = nn.LayerNorm(self.out_depth)

    def forward(
        self,
        inputs,
        state,
        fb_input,
        res_input,
        curr_timestep=None,
        # **training_kwargs
    ):
        """
        Produce outputs of RecipCell, given inputs and previous state {'cell':cell_state, 'out':out_state}

        inputs: (bs, C, H, W) dict w keys ('ff', 'fb'). ff and fb inputs must have the same shape.
        """
        # https://github.com/neuroailab/convrnns/blob/b76a44d86ca9ab2c90821d0a5f281330188f699e/convrnns/utils/cells.py#L2537

        if state is None:
            bs, _, H, W = inputs.shape
            if self.use_cell:
                state = torch.zeros(
                    bs, self.cell_depth + self.out_depth, H, W, device=inputs.device
                )
            else:
                state = torch.zeros(bs, self.out_depth, H, W, device=inputs.device)

        if self.use_cell:
            prev_cell, prev_out = torch.split(
                state, split_size_or_sections=[self.cell_depth, self.out_depth], dim=1
            )
        else:
            prev_out = state

        if self.feedback_entry == "input" and fb_input is not None:
            inputs = inputs + self.feedback_conv(fb_input, curr_timestep=curr_timestep)

        inputs = self._input_activation(inputs)

        if self.use_cell:
            cell_input = 0
            assert self.cell_residual or self.input_to_cell
            if self.cell_residual:
                assert res_input is not None
            if res_input is not None and self.cell_residual:
                cell_input = cell_input + self.res2cell_conv(
                    res_input, curr_timestep=curr_timestep
                )

            if self.input_to_cell:
                cell_input = cell_input + self.inp2cell_conv(
                    inputs, curr_timestep=curr_timestep
                )

            if fb_input is not None and self.feedback_entry == "cell":
                cell_input = cell_input + self.feedback_conv(
                    fb_input, curr_timestep=curr_timestep
                )

            cell_tau = self.cell_tau_conv(prev_cell, curr_timestep=curr_timestep)
            cell_gate = self.cell_gate_conv(prev_out, curr_timestep=curr_timestep)

            next_cell = (self._tau_offset + self._tau_k * cell_tau) * prev_cell + (
                self._gate_offset + self._gate_k * cell_gate
            ) * cell_input

            if self.layernorm:
                next_cell = self.ln_cell(next_cell.permute(0, 2, 3, 1)).permute(
                    0, 3, 1, 2
                )

            next_cell = self._cell_activation(next_cell)

        if self.input_to_out:
            out_input = self.inp2out_conv(inputs, curr_timestep=curr_timestep)
        else:
            out_input = inputs

        if self.cell_to_out and self.use_cell:
            out_input = out_input + self.cell2out_conv(
                prev_cell, curr_timestep=curr_timestep
            )

        if res_input is not None and self.out_residual:
            out_input = out_input + self.res2out_conv(
                res_input, curr_timestep=curr_timestep
            )

        if fb_input is not None and self.feedback_entry == "out":
            out_input = out_input + self.feedback_conv(
                fb_input, curr_timestep=curr_timestep
            )

        out_tau = self.out_tau_conv(prev_out, curr_timestep=curr_timestep)

        if self.use_cell and not self.cell_to_out:
            out_gate = self.cell2out_conv(prev_cell, curr_timestep=curr_timestep)
        else:
            out_gate = 0

        if res_input is not None and self.residual_to_out_gate:
            out_gate = out_gate + self.out_gate_conv(
                res_input, curr_timestep=curr_timestep
            )

        out_tau = self._tau_nonlinearity(out_tau)
        out_gate = self._gate_nonlinearity(out_gate)

        next_out = (self._tau_offset + self._tau_k * out_tau) * prev_out + (
            self._gate_offset + self._gate_k * out_gate
        ) * out_input

        if self.layernorm:
            next_out = self.ln_out(next_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        next_out = self._out_activation(next_out)

        if self.use_cell:
            next_state = torch.concat(
                [next_cell, next_out], dim=1
            )  # concat over channels
        else:
            next_state = next_out

        return next_out, next_state
