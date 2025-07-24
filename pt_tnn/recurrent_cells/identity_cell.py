from .recurrent_cell_base import RecurrentCellBase

__all__ = ["IdentityCell"]


class IdentityCell(RecurrentCellBase):
    """
    Definition for the "identity" cell. This is useful for the output node of the
    graph.
    """

    def __init__(self, state_shape=None, input_in_channels=None, **kwargs):
        super(IdentityCell, self).__init__(
            input_in_channels=input_in_channels,
            state_shape=state_shape,
            out_channels=input_in_channels,
        )

    def forward(self, inputs, state, **kwargs):
        return inputs, state
