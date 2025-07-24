import os

import torch
import torch.optim as optim

torch.manual_seed(0)

from pt_tnn import TemporalGraph

from functools import partial


def _store_features(features, layer, inp, out):
    out, state = out
    features.append(out.detach().clone())


def test_basic(config_file_path):
    print("======= Testing basic =======")
    assert os.path.isfile(config_file_path)

    tg = TemporalGraph(config_file_path)

    for n in tg.graph.nodes:
        attrs = tg.graph.nodes[n]
        print(attrs)
        print(list(tg.graph.predecessors(n)))


def test_shape_determination(config_file_path):
    print("======= Testing shape determination =======")
    assert os.path.isfile(config_file_path)

    tg = TemporalGraph(config_file_path)

    for node, attr in tg.graph.nodes(data=True):
        print(node)
        print(attr)

    for node, attr in tg.graph.nodes(data=True):
        print(node)
        print(getattr(tg, node))


def test_unroll(config_file_path):
    print("======= Testing unroll =======")
    assert os.path.isfile(config_file_path)

    tg = TemporalGraph(config_file_path)
    for node, attr in tg.graph.nodes(data=True):
        print(node)
        print(getattr(tg, node))

    N, T = 3, 2
    C, H, W = tg.input_shape
    inputs = torch.ones(N, T, C, H, W)
    output = tg(inputs, n_times=T)

    print(output.shape)

    print_modules(tg)


def test_full_unroll(config_file_path):  # Yuchen: added here
    print("======= Testing full unroll =======")
    print(config_file_path)
    assert os.path.isfile(config_file_path)

    tg = TemporalGraph(config_file_path)
    for node, attr in tg.graph.nodes(data=True):
        print(node)
        print(getattr(tg, node))

    N, T = 3, 2
    C, H, W = tg.input_shape
    inputs = torch.ones(N, T, C, H, W)
    output = tg.full_unroll(inputs, n_times=T)

    print(output.shape)

    print_modules(tg)


def print_modules(graph):
    for node, attr in graph.graph.nodes(data=True):
        sd = getattr(graph, node).state_dict()
        print(node, sd.keys())


def test_backward(config_file_path, do_batchnorm=False):
    print("======= Testing backward =======")
    assert os.path.isfile(config_file_path)

    tg = TemporalGraph(config_file_path).cuda()

    N = 3 if do_batchnorm else 1
    T = 5
    C, H, W = tg.input_shape
    inputs = torch.ones(N, T, C, H, W).cuda()

    for mod in tg.state_dict().keys():
        print(f"Module {mod}")

    # Optimizer
    opt = optim.Adam(list(tg.parameters()), lr=0.0005)

    for i in range(150):
        opt.zero_grad()
        avg = torch.mean(tg(inputs, n_times=T, cuda=True))

        loss = torch.square(avg - 100)
        print(f"[Step {i}] loss: {loss:.3f}; avg: {avg:.3f}")

        loss.backward()
        opt.step()


def test_state_dict(config_file_path):
    print("======= Testing state dict =======")
    assert os.path.isfile(config_file_path)

    tg = TemporalGraph(config_file_path)
    sd = tg.state_dict()
    for k, v in sd.items():
        print(f"{k}: {v.shape}")


def test_identity_math(config_file_path):
    assert os.path.isfile(config_file_path)
    if not config_file_path.split("/")[-1] == "test_identity_feedback.json":
        print("======= Not testing test_identity_math(...) =======")
        return
    print("======= Testing identity math =======")

    tg = TemporalGraph(config_file_path)
    assert (
        len(tg.nodes) == 2
    ), "Make sure test config file only has two nodes named 'conv1' and 'output'."

    N, T = 1, 4
    C, H, W = tg.input_shape
    inputs = torch.ones(N, T, C, H, W)

    features = list()
    out_features = list()
    handle_conv1 = tg.conv1.register_forward_hook(partial(_store_features, features))
    handle_output = tg.output.register_forward_hook(
        partial(_store_features, out_features)
    )

    outputs = tg(inputs, n_times=T, cuda=False)
    handle_conv1.remove()
    handle_output.remove()

    conv_out = None
    output_out = None
    assert len(features) == T
    assert len(out_features) == T
    for i, feat, out_feat in zip(range(T), features, out_features):
        if output_out is None:
            output_out = torch.zeros(N, 1, H, W)

        new_conv_out, new_c_state = tg.conv1(
            inputs=[inputs[:, i, :, :, :], output_out],
            input_types=["ff_input", "fb_input"],
            state=None if i == 0 else new_c_state,
        )

        if conv_out is None:
            conv_out = torch.zeros_like(new_conv_out)

        new_output_out, new_output_state = tg.output(
            inputs=[conv_out],
            input_types=["ff_input"],
            state=None if i == 0 else new_output_state,
        )

        assert torch.equal(new_conv_out, feat)
        assert torch.equal(new_output_out, out_feat)

        conv_out = new_conv_out
        output_out = new_output_out

    print("Auto:", outputs)
    print("Manual:", output_out)
    assert torch.equal(outputs, output_out)
    assert len(features) == T and len(out_features) == T


def test_feedforward(config_file_path):
    assert os.path.isfile(config_file_path)
    if not config_file_path.split("/")[-1] == "test_feedforward.json":
        print("======= Not testing test_feedforward(...) =======")
        return
    print("======= Testing feedforward =======")

    tg = TemporalGraph(config_file_path)
    assert (
        len(tg.nodes) == 2
    ), "Make sure test config file only has two nodes named 'conv1' and 'output'."

    N, T = 1, 5
    C, H, W = tg.input_shape
    inputs = torch.ones(N, T, C, H, W)

    features = list()
    handle = tg.conv1.register_forward_hook(partial(_store_features, features))
    outputs = tg(inputs, n_times=T, cuda=False)
    handle.remove()

    new_conv_out, new_c_state = tg.conv1(
        inputs=[inputs[:, 0, :, :, :]],
        input_types=["ff_input"],
        state=None,
    )
    assert torch.equal(new_conv_out, features[0])

    new_output_out, new_output_state = tg.output(
        inputs=[new_conv_out],
        input_types=["ff_input"],
        state=None,
    )

    print("Auto:", outputs)
    print("Manual:", new_output_out)
    assert torch.equal(outputs, new_output_out)

    for f in features:
        print(id(f))
        assert torch.equal(f, new_conv_out)

    assert len(features) == T


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--do-batchnorm", action="store_true")
    parser.set_defaults(do_batchnorm=False)
    args = parser.parse_args()

    print(f"Doing batchnorm? {args.do_batchnorm}")

    test_basic(args.config)
    test_shape_determination(args.config)
    test_unroll(args.config)
    test_backward(args.config, args.do_batchnorm)
    test_state_dict(args.config)
    test_identity_math(args.config)
    test_feedforward(args.config)
    # Yuchen: added here
    test_full_unroll(args.config)
