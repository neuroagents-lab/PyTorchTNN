import torch

from pt_tnn.harbor_policy import ResizeConcat

INPUT_TYPES = ["ff_input", "fb_input", "skip_input", "fb_input"]


def test_downsample():
    N = 10
    S = 4

    inputs = list()
    inputs.append(torch.rand(N, 3, 15, 15))
    inputs.append(torch.rand(N, 5, 25, 25))
    inputs.append(torch.rand(N, 2, 5, 5))
    inputs.append(torch.rand(N, 1, 13, 13))

    resize_concat = ResizeConcat(input_shape=(S, S), in_channels=[3, 5, 2, 1])
    resize_concat.initialize_operations([inp.shape[1:] for inp in inputs])

    outputs = resize_concat(inputs, INPUT_TYPES)
    print(outputs.shape)
    assert outputs.shape == (N, 11, S, S)


def test_upsample():
    N = 10
    S = 30

    inputs = list()
    inputs.append(torch.rand(N, 3, 15, 15))
    inputs.append(torch.rand(N, 5, 25, 25))
    inputs.append(torch.rand(N, 2, 5, 5))
    inputs.append(torch.rand(N, 1, 13, 13))

    resize_concat = ResizeConcat(input_shape=(S, S), in_channels=[3, 5, 2, 1])
    resize_concat.initialize_operations([inp.shape[1:] for inp in inputs])

    outputs = resize_concat(inputs, INPUT_TYPES)
    print(outputs.shape)
    assert outputs.shape == (N, 11, S, S)


def test_rectangular():
    N = 10
    S = 30

    inputs = list()
    inputs.append(torch.rand(N, 3, 15, 15))
    inputs.append(torch.rand(N, 5, 25, 40))
    inputs.append(torch.rand(N, 2, 5, 10))
    inputs.append(torch.rand(N, 1, 10, 13))

    resize_concat = ResizeConcat(input_shape=(S, S), in_channels=[3, 5, 2, 1])
    resize_concat.initialize_operations([inp.shape[1:] for inp in inputs])

    outputs = resize_concat(inputs, INPUT_TYPES)
    print(outputs.shape)
    assert outputs.shape == (N, 11, S, S)


def test_rectangular_output():
    N = 10
    H = 30
    W = 20

    inputs = list()
    inputs.append(torch.rand(N, 3, 15, 15))
    inputs.append(torch.rand(N, 5, 25, 40))
    inputs.append(torch.rand(N, 2, 5, 10))
    inputs.append(torch.rand(N, 1, 10, 13))

    resize_concat = ResizeConcat(input_shape=(H, W), in_channels=[3, 5, 2, 1])
    resize_concat.initialize_operations([inp.shape[1:] for inp in inputs])

    outputs = resize_concat(inputs, INPUT_TYPES)
    print(outputs.shape)
    assert outputs.shape == (N, 11, H, W)


if __name__ == "__main__":
    test_downsample()
    test_upsample()
    test_rectangular()
    test_rectangular_output()
