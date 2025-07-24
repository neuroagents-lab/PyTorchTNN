import torch.nn as nn

from . import harbor_policy as hp
from . import pre_post_memory as pre_post_mem
from . import recurrent_cells as r_cells

__all__ = ["RecurrentModule", "RecurrentModuleRGC"]


class RecurrentModule(nn.Module):
    """
    Class definition of the recurrent module for each node in the computation graph.
    Arguments for each module within this recurrent module are defined by their
    respective classes.

    Args:
        harbor_policy_args (dict): keyword arguments for the desired harbor policy
        pre_memory_args (dict): keyword arguments for the desired pre memory module
        recurrent_cell_args (dict): keyword arguments for the recurrent cell
        post_memory_args (dict): keyword arguments for the post memory module
        module_attr (dict): attributes of the current module including it's name,
            number of input and output channels, etc. See `pt_tnn/temporal_graph.py`.
        num_timesteps (int or None): number of timesteps to unroll the temporal graph.
    """

    def __init__(
        self,
        harbor_policy_args,
        pre_memory_args,
        recurrent_cell_args,
        post_memory_args,
        module_attr,
        num_timesteps,
    ):
        super(RecurrentModule, self).__init__()

        self._construct_recurrent_module(
            harbor_policy_args=harbor_policy_args,
            pre_memory_args=pre_memory_args,
            recurrent_cell_args=recurrent_cell_args,
            post_memory_args=post_memory_args,
            module_attr=module_attr,
            num_timesteps=num_timesteps,
        )

        self.residual = module_attr["residual"]

    @property
    def harbor_policy(self):
        return self._harbor_policy

    @property
    def pre_memory(self):
        return self._pre_memory

    @property
    def recurrent_cell(self):
        return self._recurrent_cell

    @property
    def post_memory(self):
        return self._post_memory

    @property
    def name(self):
        return self._name

    def _init_module(self, args, lib):
        assert "name" in args
        module_name = args.pop("name")
        module = getattr(lib, module_name)(**args)
        return module

    def _construct_recurrent_module(
        self,
        harbor_policy_args,
        pre_memory_args,
        recurrent_cell_args,
        post_memory_args,
        module_attr,
        num_timesteps,
    ):
        # Harbor policy
        harbor_policy_args["input_shape"] = module_attr["input_shape"]
        harbor_policy_args["in_channels"] = module_attr["in_channels"]
        self._harbor_policy = self._init_module(harbor_policy_args, hp)

        # Pre-memory
        pre_memory_args["in_channels"] = self._harbor_policy.out_channels
        pre_memory_args["num_timesteps"] = num_timesteps
        self._pre_memory = self._init_module(pre_memory_args, pre_post_mem)

        # Recurrent cell
        recurrent_cell_args["input_in_channels"] = self._pre_memory.out_channels
        self._recurrent_cell = self._init_module(recurrent_cell_args, r_cells)

        # Post-memory
        post_memory_args["num_timesteps"] = num_timesteps
        self._post_memory = self._init_module(post_memory_args, pre_post_mem)

        # Recurrent module name
        self._name = module_attr["name"]

    def _check_input_shapes(self, inputs):
        # Using `1:` because we are omitting batch dimension
        curr_input_shapes = [tuple(i.shape[1:]) for i in inputs]
        assert set(curr_input_shapes) == set(self._harbor_policy.input_shapes)

    def forward(self, inputs=None, input_types=None, state=None, curr_timestep=None):
        if inputs is not None:
            self._check_input_shapes(inputs)

        # Combine the inputs for the current module. If the harbor_policy
        # is None, this means that it has not been initialized yet.
        if (
            self._harbor_policy.operations_initialized
        ):  # called in training / validation
            harbor_output = self._harbor_policy(inputs, input_types)
        else:  # only called in the creation of TemporalGraph; thus, always on cpu
            harbor_output = self._harbor_policy.random_input()

        # Apply the module's pre-memory operation (e.g., convolution)
        pre_mem_output = self._pre_memory(harbor_output, curr_timestep=curr_timestep)

        # Apply the module's recurrent operation to update the memory
        recurrent_output, state = self._recurrent_cell(pre_mem_output, state)

        # Apply the module's post-memory operation (e.g., maxpool)
        if self.residual:
            output = self._post_memory(
                recurrent_output, residual=harbor_output, curr_timestep=curr_timestep
            )
        else:
            output = self._post_memory(recurrent_output, curr_timestep=curr_timestep)
        return output, state


class RecurrentModuleRGC(RecurrentModule):
    """
    Class definition of the recurrent module for each node in the computation graph.
    Arguments for each module within this recurrent module are defined by their
    respective classes.

    Args:
        harbor_policy_args (dict): keyword arguments for the desired harbor policy
        pre_memory_args (dict): keyword arguments for the desired pre memory module
        recurrent_cell_args (dict): keyword arguments for the recurrent cell
        post_memory_args (dict): keyword arguments for the post memory module
        module_attr (dict): attributes of the current module including it's name,
            number of input and output channels, etc. See `pt_tnn/temporal_graph.py`.
        num_timesteps (int or None): number of timesteps to unroll the temporal graph.
    """

    def __init__(
        self,
        harbor_policy_args,
        pre_memory_args,
        recurrent_cell_args,
        post_memory_args,
        module_attr,
        num_timesteps,
    ):
        super(RecurrentModuleRGC, self).__init__(
            harbor_policy_args=harbor_policy_args,
            pre_memory_args=pre_memory_args,
            recurrent_cell_args=recurrent_cell_args,
            post_memory_args=post_memory_args,
            module_attr=module_attr,
            num_timesteps=num_timesteps,
        )
        self.ff_channels = module_attr[
            "ff_channels"
        ]  # specified in temporal_graph.py `_setup_graph_attrs`
        if len(self._harbor_policy._in_channels) == 1:
            self.has_fb = False
            assert self.ff_channels == self._harbor_policy._in_channels[0]
        else:
            self.has_fb = True
            assert self.ff_channels in self._harbor_policy._in_channels

    def _construct_recurrent_module(
        self,
        harbor_policy_args,
        pre_memory_args,
        recurrent_cell_args,
        post_memory_args,
        module_attr,
        num_timesteps,
    ):
        # Harbor policy
        harbor_policy_args["input_shape"] = module_attr["input_shape"]
        harbor_policy_args["in_channels"] = module_attr["in_channels"]
        self._harbor_policy = self._init_module(harbor_policy_args, hp)

        # Pre-memory
        if len(self._harbor_policy._in_channels) == 1:  # has no feedback connection
            pre_memory_args["in_channels"] = self._harbor_policy.out_channels
        else:  # has feedback connections
            pre_memory_args["in_channels"] = module_attr["ff_channels"]

        pre_memory_args["num_timesteps"] = num_timesteps
        self._pre_memory = self._init_module(pre_memory_args, pre_post_mem)

        # Recurrent cell
        recurrent_cell_args["input_in_channels"] = self._pre_memory.out_channels
        recurrent_cell_args["res_channels"] = module_attr["ff_channels"]
        if len(self._harbor_policy._in_channels) == 1:  # has no feedback connection
            recurrent_cell_args["fb_channels"] = None
        else:  # has feedback connections
            recurrent_cell_args["fb_channels"] = (
                sum(module_attr["in_channels"]) - module_attr["ff_channels"]
            )
        self._recurrent_cell = self._init_module(recurrent_cell_args, r_cells)

        # Post-memory
        post_memory_args["num_timesteps"] = num_timesteps
        self._post_memory = self._init_module(post_memory_args, pre_post_mem)

        # Recurrent module name
        self._name = module_attr["name"]

    def forward(self, inputs=None, input_types=None, state=None, curr_timestep=None):
        # https://github.com/neuroailab/convrnns/blob/b76a44d86ca9ab2c90821d0a5f281330188f699e/convrnns/utils/cells.py#L2537
        if inputs is not None:
            self._check_input_shapes(inputs)

        # Combine the inputs for the current module. If the harbor_policy
        # is None, this means that it has not been initialized yet.
        if (
            self._harbor_policy.operations_initialized
        ):  # called in training / validation
            harbor_output = self._harbor_policy(inputs, input_types)
        else:  # only called in the creation of TemporalGraph; thus, always on cpu
            harbor_output = self._harbor_policy.random_input()

        if self.has_fb:  # fb_input will have its own non-zero channels
            harbor_output, fb_input = (
                harbor_output[:, : self.ff_channels, :, :],
                harbor_output[:, self.ff_channels :, :, :],
            )
        else:
            fb_input = None  # no fb_input, no need to split the harbor_output

        # Apply the module's pre-memory operation (e.g., convolution)
        pre_mem_output = self._pre_memory(harbor_output, curr_timestep=curr_timestep)

        # Apply the module's recurrent operation to update the memory
        recurrent_output, state = self._recurrent_cell(
            pre_mem_output,
            state,
            fb_input=fb_input,
            res_input=harbor_output,
            curr_timestep=curr_timestep,
        )

        # Apply the module's post-memory operation (e.g., maxpool)
        if self.residual:
            output = self._post_memory(
                recurrent_output, residual=harbor_output, curr_timestep=curr_timestep
            )
        else:
            output = self._post_memory(recurrent_output, curr_timestep=curr_timestep)
        return output, state
