import torch

from pt_tnn.harbor_policy import Identity

INPUT_TYPES = ["ff_input", "fb_input", "skip_input", "fb_input"]


def test_basic():
    N = 10
    S = 4

    inputs = list()
    inputs.append(torch.rand(N, 3, S, S))
    inputs.append(torch.rand(N, 5, S, S))
    inputs.append(torch.rand(N, 2, S, S))
    inputs.append(torch.rand(N, 1, S, S))

    identity = Identity(input_shape=(S, S), in_channels=[3, 5, 2, 1])
    identity.initialize_operations([inp.shape[1:] for inp in inputs])

    outputs = identity(inputs, INPUT_TYPES)
    print(outputs.shape)
    assert outputs.shape == (N, 11, S, S)


def test_different_size():
    N = 10
    S = 4

    inputs = list()
    inputs.append(torch.rand(N, 3, S, S))
    inputs.append(torch.rand(N, 5, S, 8))
    inputs.append(torch.rand(N, 2, S, S))
    inputs.append(torch.rand(N, 1, S, S))

    identity = Identity(input_shape=(S, S), in_channels=[3, 5, 2, 1])
    identity.initialize_operations([inp.shape[1:] for inp in inputs])

    outputs = identity(inputs, INPUT_TYPES)
    print(outputs.shape)
    assert outputs.shape == (N, 11, S, S)


if __name__ == "__main__":
    test_basic()
    test_different_size()
