import torch

from pt_tnn.recurrent_cells import IdentityCell


def test_basic():
    N = 10
    inputs = torch.rand(N, 3, 15, 15)
    state = torch.rand(N, 11, 15, 15)

    convrnn_cell = IdentityCell(input_in_channels=3)

    output, state = convrnn_cell(inputs, state)
    print(output.shape)
    assert torch.equal(output, inputs)


if __name__ == "__main__":
    test_basic()
