import torch

from pt_tnn.pre_post_memory import ActivationLayer


def test_basic():
    N = 17
    inputs = torch.rand(N, 11, 15, 15)

    activation = ActivationLayer(activation="ReLU")

    outputs = activation(inputs)
    print(outputs.shape)
    assert outputs.shape == (N, 11, 15, 15)
    assert torch.equal(outputs, inputs.clamp_min(0))


if __name__ == "__main__":
    test_basic()
