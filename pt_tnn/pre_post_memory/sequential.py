import torch.nn as nn
import pt_tnn.pre_post_memory as pre_post_mem
from .memory_base import MemoryBase

__all__ = ["Sequential"]


class Sequential(MemoryBase):
    """
    Class to instantiate the "sequential" operation that combines a sequence of operations.
    """

    def __init__(self, list_of_func_kwargs: list[dict], **kwargs):
        in_channels = list_of_func_kwargs[0].get("in_channels", None)
        out_channels = list_of_func_kwargs[-1].get("out_channels", None)
        num_timesteps = kwargs.get("num_timesteps", None)

        super(Sequential, self).__init__(
            in_channels=in_channels, out_channels=out_channels
        )

        module_list = []
        for func_kwargs in list_of_func_kwargs:
            func_kwargs.update(
                {"num_timesteps": num_timesteps}
            )  # add num_timesteps to support time varying batchnorm
            module_list.append(self._init_module(args=func_kwargs, lib=pre_post_mem))
        self.sequential_mem = nn.ModuleList(module_list)
        # register as model parameters

    @staticmethod
    def _init_module(args, lib):
        assert "name" in args
        module_name = args.pop("name")
        module = getattr(lib, module_name)(**args)
        return module

    def forward(self, x, curr_timestep, **kwargs):
        for mem in self.sequential_mem:
            x = mem(x, curr_timestep=curr_timestep, **kwargs)
        return x
