import torch

from pt_tnn.recurrent_cells import TimeDecayRecurrentCell


def test_basic():
    N = 10
    inputs = torch.rand(N, 3, 15, 15)
    state = torch.rand(N, 3, 15, 15)

    convrnn_cell = TimeDecayRecurrentCell(input_in_channels=3, tau=0.1)

    output, output_state = convrnn_cell(inputs, state)
    print(output.shape)
    assert torch.equal(output, inputs + 0.1 * state)


if __name__ == "__main__":
    test_basic()
