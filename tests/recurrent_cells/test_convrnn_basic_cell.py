import torch

from pt_tnn.recurrent_cells import ConvRNNBasicCell


def test_basic():
    N = 10
    inputs = torch.rand(N, 3, 15, 15)
    state = torch.rand(N, 20, 15, 15)

    convrnn_cell = ConvRNNBasicCell(
        input_in_channels=3,
        out_channels=20,
        state_shape=(15, 15),
        activation="ReLU",
        ksize=(3, 3),
        stride=1,
        padding=1,
    )

    output, state = convrnn_cell(inputs, state)
    print(output.shape, state.shape)
    assert output.shape == (N, 20, 15, 15)
    assert state.shape == (N, 20, 15, 15)


def _test_math(N=1, multiplier=1, activation="ReLU", activation_func_kwargs={}):
    inputs = torch.ones(N, 3, 3, 3) * multiplier
    state = torch.ones(N, 2, 3, 3) * multiplier

    convrnn_cell = ConvRNNBasicCell(
        input_in_channels=3,
        out_channels=1,
        state_shape=(3, 3),
        activation=activation,
        ksize=(3, 3),
        stride=1,
        padding=0,
        activation_func_kwargs=activation_func_kwargs,
    )

    convrnn_cell.conv_input.weight.data = torch.ones(1, 3, 3, 3)
    convrnn_cell.conv_input.bias.data = torch.zeros(1)

    convrnn_cell.conv_state.weight.data = torch.ones(1, 2, 3, 3)
    convrnn_cell.conv_state.bias.data = torch.zeros(1)

    output, state = convrnn_cell(inputs, state)
    print(output.shape, state.shape)
    print(output.data, state.data)

    return output, state


def test_math_positive():
    N = 1
    output, state = _test_math(N=N, multiplier=1)

    assert output.shape == (N, 1, 1, 1)
    assert state.shape == (N, 1, 1, 1)
    assert output.item() == 45
    assert state.item() == 45


def test_math_negative():
    N = 1
    output, state = _test_math(N=N, multiplier=-1)

    assert output.shape == (N, 1, 1, 1)
    assert state.shape == (N, 1, 1, 1)
    assert output.item() == 0
    assert state.item() == 0


def test_activation_func_kwargs():
    N = 1
    af = {"negative_slope": 0.1}

    output, state = _test_math(
        N=1, multiplier=-1, activation="LeakyReLU", activation_func_kwargs=af
    )

    assert output.shape == (N, 1, 1, 1)
    assert state.shape == (N, 1, 1, 1)
    assert output.item() == -45 * af["negative_slope"]
    assert state.item() == -45 * af["negative_slope"]


def test_zero_state():
    N = 1
    inputs = torch.ones(N, 3, 3, 3)
    state = torch.zeros(N, 2, 3, 3)

    convrnn_cell = ConvRNNBasicCell(
        input_in_channels=3,
        out_channels=1,
        state_shape=(3, 3),
        activation="ReLU",
        ksize=(3, 3),
        stride=1,
        padding=0,
        activation_func_kwargs={},
    )

    convrnn_cell.conv_input.weight.data = torch.ones(1, 3, 3, 3)
    convrnn_cell.conv_input.bias.data = torch.zeros(1)

    convrnn_cell.conv_state.weight.data = torch.ones(1, 2, 3, 3)
    convrnn_cell.conv_state.bias.data = torch.zeros(1)

    output, state = convrnn_cell(inputs, state)
    print(output.shape, state.shape)
    assert output.item() == 27


if __name__ == "__main__":
    test_basic()
    test_math_positive()
    test_math_negative()
    test_activation_func_kwargs()
    test_zero_state()
