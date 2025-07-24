import pt_tnn.pre_post_memory as pre_post_mem
from .memory_base import MemoryBase
import torch.nn as nn

__all__ = ["ResidualAdd"]


class ResidualAdd(MemoryBase):
    """
    Class to instantiate the "add residual" operation.
    """

    def __init__(
        self,
        feedforward_func_kwargs: dict,
        residual_func_kwargs: dict = None,
        # parameters for activation
        activation=None,
        activation_func_kwargs={},
        **kwargs
    ):
        in_channels = feedforward_func_kwargs.get("in_channels", None)
        out_channels = feedforward_func_kwargs.get("out_channels", None)
        num_timesteps = kwargs.get("num_timesteps", None)

        super(ResidualAdd, self).__init__(
            in_channels=in_channels, out_channels=out_channels
        )

        feedforward_func_kwargs.update({"num_timesteps": num_timesteps})
        # add num_timesteps to support time varying batchnorm
        self.feedforward = self._init_module(
            args=feedforward_func_kwargs, lib=pre_post_mem
        )

        if residual_func_kwargs is None:
            self.residual = pre_post_mem.Identity()
        else:
            residual_func_kwargs.update({"num_timesteps": num_timesteps})
            # add num_timesteps to support time varying batchnorm
            self.residual = self._init_module(
                args=residual_func_kwargs, lib=pre_post_mem
            )

        self.activation = (
            getattr(nn, activation)(**activation_func_kwargs)
            if activation is not None
            else nn.Identity()
        )

    @staticmethod
    def _init_module(args, lib):
        assert "name" in args
        module_name = args.pop("name")
        module = getattr(lib, module_name)(**args)
        return module

    def forward(self, x, residual, curr_timestep, **kwargs):
        output = self.feedforward(x, curr_timestep, **kwargs) + self.residual(
            residual, curr_timestep, **kwargs
        )
        return self.activation(output)
