import torch

from pt_tnn.pre_post_memory import AdaptiveAvgPool


def test_downsample():
    N = 10
    inputs = torch.rand(N, 11, 15, 15)

    pool = AdaptiveAvgPool(output_size=(3, 3))

    outputs = pool(inputs)
    print(outputs.shape)
    assert outputs.shape == (N, 11, 3, 3)


def test_upsample():
    N = 10
    inputs = torch.rand(N, 11, 15, 15)

    pool = AdaptiveAvgPool(output_size=(20, 20))

    outputs = pool(inputs)
    print(outputs.shape)
    assert outputs.shape == (N, 11, 20, 20)


if __name__ == "__main__":
    test_downsample()
    test_upsample()
