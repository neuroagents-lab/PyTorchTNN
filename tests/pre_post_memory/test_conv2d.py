import torch

from pt_tnn.pre_post_memory import Conv2dCell


def test_basic():
    N = 10
    inputs = torch.rand(N, 11, 15, 15)

    conv2d = Conv2dCell(
        in_channels=11, out_channels=20, ksize=(3, 3), stride=1, padding=1
    )

    outputs = conv2d(inputs)
    print(outputs.shape)
    assert outputs.shape == (N, 20, 15, 15)


if __name__ == "__main__":
    test_basic()
