import torch

from .memory_base import MemoryBase
import torch.nn as nn
import torch.nn.init as init

__all__ = ["Conv2dCell"]


class Conv2dCell(MemoryBase):
    """
    Defines the 2d convolution operation that is applied on the current input
    that can be used in both pre-memory & post-memory.

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
        padding=0,
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
        super(Conv2dCell, self).__init__(
            in_channels=in_channels, out_channels=out_channels
        )

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
