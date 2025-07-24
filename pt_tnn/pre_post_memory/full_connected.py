import torch
import torch.nn as nn
from .memory_base import MemoryBase
from pt_tnn.utils.lazy_layer import LazyLinearWithInit


class FullyConnected(MemoryBase):
    """
    Definition for the linear layers. This is useful for the output node of the
    graph.
    """

    def __init__(
        self,
        # LazyLinear arguments
        in_channels=None,
        out_channels=None,
        # parameters for kernel_init
        init_dict=None,
        # parameters for bias_init
        use_bias=True,
        bias=None,
        # dropout, batchnorm, activation arguments
        dropout=0.0,
        batchnorm=None,
        batchnorm_timevary=False,
        # parameters for activation
        activation="ReLU",
        activation_func_kwargs={},
        num_timesteps=None,
        **kwargs,
    ):
        super(FullyConnected, self).__init__(
            in_channels=in_channels, out_channels=out_channels
        )
        self.lin = LazyLinearWithInit(
            out_features=out_channels,
            init_dict=init_dict,
            use_bias=use_bias,
            bias_val=bias,
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.num_timesteps = num_timesteps
        self.batchnorm_timevary = batchnorm_timevary
        if batchnorm_timevary:
            print(f"using time-dependent BatchNorm for {num_timesteps} timesteps...")
            assert num_timesteps is not None

            self.bns = nn.ModuleList(
                [
                    nn.LazyBatchNorm1d() if batchnorm is not None else nn.Identity()
                    for _ in range(num_timesteps)
                ]
            )
        else:
            self.bn = nn.LazyBatchNorm1d() if batchnorm is not None else nn.Identity()

        self.activation = (
            getattr(nn, activation)(**activation_func_kwargs)
            if activation is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, curr_timestep=None, **kwargs):
        x = x.reshape(x.size(0), -1)  # flatten
        if not self.batchnorm_timevary:
            curr_bn = self.bn
        else:
            assert curr_timestep is not None
            curr_bn = self.bns[curr_timestep]

        output = curr_bn(self.lin(self.dropout(x)))

        return self.activation(output)
