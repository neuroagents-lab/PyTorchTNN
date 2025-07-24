import torch

from pt_tnn.pre_post_memory import MaxPool


def test_basic():
    N = 10
    inputs = torch.rand(N, 11, 15, 15)

    maxpool = MaxPool(ksize=(3, 3), stride=1, padding=1)

    outputs = maxpool(inputs)
    print(outputs.shape)
    assert outputs.shape == (N, 11, 15, 15)


def test_downsample():
    N = 10
    inputs = torch.rand(N, 11, 15, 15)

    maxpool = MaxPool(ksize=(3, 3), stride=2, padding=1)

    outputs = maxpool(inputs)
    print(outputs.shape)
    assert outputs.shape == (N, 11, 8, 8)


if __name__ == "__main__":
    test_basic()
    test_downsample()
