import torch

from pt_tnn.pre_post_memory import FullyConnected


def test_basic():
    N = 17
    inputs = torch.rand(N, 11, 15, 15)

    linear = FullyConnected(
        out_channels=10, init_dict={"method": "zeros_"}, use_bias=True, bias=0.1
    )

    outputs = linear(inputs)
    print(outputs.shape)
    print(outputs.sum())
    assert outputs.shape == (N, 10)
    assert torch.allclose(outputs.sum(), torch.tensor(N, dtype=torch.float))


if __name__ == "__main__":
    test_basic()
