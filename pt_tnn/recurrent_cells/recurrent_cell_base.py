import torch
import torch.nn as nn

__all__ = ["RecurrentCellBase"]


class RecurrentCellBase(nn.Module):
    def __init__(self, input_in_channels=None, state_shape=None, out_channels=None):
        super(RecurrentCellBase, self).__init__()

        assert input_in_channels is not None
        assert out_channels is not None

        self._input_in_channels = input_in_channels
        self._state_shape = state_shape
        self._out_channels = out_channels

    @property
    def input_in_channels(self):
        return self._input_in_channels

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def state_in_channels(self):
        # The number of in-channels for the state is the same as the number of output
        # channels of the recurrent cell.
        return self._out_channels

    def zero_state(self, batch_size, device):
        assert self.state_shape is not None
        assert self.state_in_channels is not None

        assert (
            len(self.state_shape) == 2
        ), "State shape must be length 2 for height and width."

        state = torch.zeros(
            batch_size,
            self.state_in_channels,
            self.state_shape[0],
            self.state_shape[1],
            device=device,
        )

        return state

    def forward(self, inputs, state, **kwargs):
        raise NotImplementedError
