import torch

from pt_tnn.pre_post_memory import ResidualAdd


def test_identity_residual():
    N = 10
    inputs = torch.rand(N, 20, 15, 15)

    residual = ResidualAdd(
        feedforward_func_kwargs={
            "name": "Conv2dCell",
            "in_channels": 20,
            "out_channels": 20,
            "ksize": [3, 3],
            "stride": 1,
            "padding": 1,
        },
        residual_func_kwargs=None,
    )

    outputs = residual(inputs, residual=inputs, curr_timestep=None)
    assert outputs.shape == (N, 20, 15, 15)


def test_downsampling_residual():
    N = 10
    inputs = torch.rand(N, 20, 15, 15)
    residual_inputs = torch.rand(N, 11, 15, 15)

    residual = ResidualAdd(
        feedforward_func_kwargs={
            "name": "Conv2dCell",
            "in_channels": 20,
            "out_channels": 20,
            "ksize": [3, 3],
            "stride": 1,
            "padding": 1,
        },
        residual_func_kwargs={
            "name": "Conv2dCell",
            "in_channels": 11,
            "out_channels": 20,
            "ksize": [3, 3],
            "stride": 1,
            "padding": 1,
        },
    )

    outputs = residual(inputs, residual=residual_inputs, curr_timestep=None)
    assert outputs.shape == (N, 20, 15, 15)


if __name__ == "__main__":
    # test_identity_residual()
    test_downsampling_residual()
