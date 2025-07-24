import torch

from pt_tnn.pre_post_memory import Identity


def test_pre_memory_identity():
    N = 10
    C = 11
    inputs = torch.rand(N, C, 15, 15)

    identity = Identity(in_channels=C)

    outputs = identity(inputs)
    print(outputs.shape)
    assert torch.equal(inputs, outputs)


def test_post_memory_identity():
    N = 10
    inputs = torch.rand(N, 11, 15, 15)

    identity = Identity()

    outputs = identity(inputs)
    assert torch.equal(inputs, outputs)


if __name__ == "__main__":
    test_pre_memory_identity()
    test_post_memory_identity()
