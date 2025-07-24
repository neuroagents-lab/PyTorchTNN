from .recurrent_cell_base import RecurrentCellBase
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ["TimeDecayRecurrentCell"]


class TimeDecayRecurrentCell(RecurrentCellBase):
    def __init__(
        self,
        state_shape=None,
        input_in_channels=None,
        init_dict=None,
        tau=0.0,
        trainable=True,
        **kwargs
    ):
        super(TimeDecayRecurrentCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=input_in_channels,
        )
        self.tau = nn.Parameter(
            torch.tensor(tau, dtype=torch.float32), requires_grad=trainable
        )
        # default: init tau as 0.0 & trainable

        if init_dict is not None:
            init_method_name = init_dict.pop("method")
            init_method = getattr(
                init, init_method_name
            )  # retrieve the method by string
            init_method(self.tau, **init_dict)

    def forward(self, inputs, state, **kwargs):
        # should return output, state
        if state is None:
            state = torch.zeros_like(inputs, device=inputs.device)
            # we bypass the zero_state in RecurrentCellBase (which requires `state_shape` and `state_in_channels`
            # not None), because in time-decay-cell the shape of `inputs` and `state` should be the same
        new_state = inputs + self.tau * state
        return new_state, new_state
