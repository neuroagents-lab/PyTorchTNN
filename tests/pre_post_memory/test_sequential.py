import torch

from pt_tnn.pre_post_memory import Sequential


def test_basic():
    N = 17
    inputs = torch.rand(N, 11, 15, 15)

    sequential = Sequential(
        list_of_func_kwargs=[
            {
                "name": "Conv2dCell",
                "in_channels": 11,
                "out_channels": 20,
                "ksize": [3, 3],
                "stride": 1,
                "padding": 1,
            },
            {
                "name": "FullyConnected",
                "out_channels": 10,
                "init_dict": {"method": "zeros_"},
                "use_bias": True,
                "bias": 0.1,
            },
        ]
    )

    outputs = sequential(inputs, curr_timestep=None)
    print(outputs.shape)
    print(outputs.sum())
    assert outputs.shape == (N, 10)
    assert torch.allclose(outputs.sum(), torch.tensor(N, dtype=torch.float))


if __name__ == "__main__":
    test_basic()
