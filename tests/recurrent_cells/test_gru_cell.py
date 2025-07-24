import torch

from pt_tnn.recurrent_cells import GRUCell


def test_basic():
    N = 10
    inputs = torch.rand(N, 3, 15, 15)
    state = torch.rand(N, 3, 15, 15)

    convrnn_cell = GRUCell(
        input_in_channels=3, out_channels=3, ksize=[1, 1], layernorm=True
    )

    output, state = convrnn_cell(inputs, state)
    print(output.shape)
    assert output.shape == (N, 3, 15, 15)
    assert state.shape == (N, 3, 15, 15)


if __name__ == "__main__":
    test_basic()
