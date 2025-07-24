import torch

from functools import partial

from pt_tnn import RecurrentModule
from pt_tnn.harbor_policy import ResizeConcat
from pt_tnn.recurrent_cells import ConvRNNBasicCell
from pt_tnn.pre_post_memory import Conv2dCell, Identity


def _store_features(features, layer, inp, out):
    out = out.detach().cpu().numpy()
    features.append(out)


def _test_basic(
    N=10,
    S=5,
    out_channels=20,
    hook=False,
    none_state=False,
    post_memory_type="Identity",
):
    input_types = ["ff_input", "fb_input", "skip_input", "fb_input"]

    # Define inputs and the hidden state
    inputs = list()
    inputs.append(torch.rand(N, 3, 15, 15))
    inputs.append(torch.rand(N, 5, 25, 25))
    inputs.append(torch.rand(N, 2, 7, 7))
    inputs.append(torch.rand(N, 1, 13, 13))
    input_shapes = [tuple(inp.shape[1:]) for inp in inputs]
    state = torch.rand(N, out_channels, S, S)

    # Harbor policy
    harbor_policy_args = {"name": "ResizeConcat"}

    # Pre-memory operation
    pre_memory_args = {
        "name": "Conv2dCell",
        "out_channels": 10,
        "ksize": [3, 3],
        "stride": 1,
        "padding": 1,
    }

    # Recurrent cell
    recurrent_cell_args = {
        "name": "ConvRNNBasicCell",
        "state_shape": [S, S],
        "out_channels": out_channels,
        "activation": "ReLU",
        "ksize": [3, 3],
        "stride": 1,
        "padding": 1,
    }

    # Post-memory operation
    post_memory_args = {"name": post_memory_type}
    if post_memory_type == "MaxPool":
        post_memory_args.update({"ksize": (3, 3), "stride": 2, "padding": 1})

    # Module attributes
    module_attrs = {
        "name": "test",
        "in_channels": 11,
        "input_shape": [S, S],
        "residual": False,
    }

    # Recurrent module
    recurrent_module = RecurrentModule(
        harbor_policy_args=harbor_policy_args,
        pre_memory_args=pre_memory_args,
        recurrent_cell_args=recurrent_cell_args,
        post_memory_args=post_memory_args,
        module_attr=module_attrs,
        num_timesteps=None,  # fix BN across time
    )

    # Need to initialize harbor policy operation
    hp = recurrent_module.harbor_policy
    hp.initialize_operations(input_shapes)
    print(recurrent_module)
    print(recurrent_module.state_dict().keys())

    for name, layer in recurrent_module.named_children():
        layer.__name__ = name
        if "recurrent_cell" in name:
            layer.register_forward_hook(
                lambda layer, _, output: print(
                    f"{layer.__name__}: {output[0].shape, output[1].shape}"
                )
            )
        else:
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    # Hook pre-memory output?
    if hook:
        features = list()
        recurrent_module.pre_memory.register_forward_hook(
            partial(_store_features, features)
        )

    # Test the case where the state is not initialized yet
    if none_state:
        state = None

    # Feed recurrent module inputs
    output, state = recurrent_module(
        inputs=inputs, input_types=input_types, state=state
    )

    # Return intermediate features if hook was desired
    if hook:
        return output, state, features
    else:
        return output, state


def test_basic():
    print("======= Testing basic =======")
    N = 10
    S = 5
    C = 20
    output, state = _test_basic(N=N, S=S, out_channels=C)

    print(output.shape, state.shape)
    assert output.shape == (N, C, S, S)
    assert state.shape == (N, C, S, S)


def test_hook():
    print("======= Testing hook =======")
    N = 10
    S = 5
    C = 20
    output, state, features = _test_basic(N=N, S=S, out_channels=C, hook=True)

    print(output.shape, state.shape)
    assert output.shape == (N, C, S, S)
    assert state.shape == (N, C, S, S)

    assert len(features) == 1
    print(features[0].shape)
    assert features[0].shape == (N, 10, S, S)


def test_post_memory_state():
    print("======= Testing post-memory state =======")
    N = 10
    S = 5
    C = 20
    output, state, features = _test_basic(
        N=N, S=S, out_channels=C, hook=True, post_memory_type="MaxPool"
    )

    print(output.shape, state.shape)

    # Post memory operation only operates on the output and not the state
    assert output.shape == (N, C, int(S / 2) + 1, int(S / 2) + 1)
    assert state.shape == (N, C, S, S)

    assert len(features) == 1
    print(features[0].shape)
    assert features[0].shape == (N, 10, S, S)


def test_zero_state():
    print("======= Testing zero state =======")
    N = 2
    S = 5
    C = 10
    output, state, features = _test_basic(
        N=N, S=S, out_channels=C, hook=True, none_state=True
    )

    print(output.shape, state.shape)
    assert output.shape == (N, C, S, S)
    assert state.shape == (N, C, S, S)

    assert len(features) == 1
    print(features[0].shape)
    assert features[0].shape == (N, 10, S, S)


if __name__ == "__main__":
    test_basic()
    test_hook()
    test_zero_state()
    test_post_memory_state()
