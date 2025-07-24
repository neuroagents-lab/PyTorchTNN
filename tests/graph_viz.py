import argparse
import torch

from torchviz import make_dot
from pt_tnn.temporal_graph import TemporalGraph

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
args = parser.parse_args()

net = TemporalGraph(args.config)

N, T = 1, 2
C, H, W = net.input_shape
inputs = torch.ones(N, T, C, H, W)

outputs = net(inputs, n_times=T, cuda=False)
print(outputs.shape)

for n, p in net.named_parameters():
    print(n, p.shape)

print(net)

make_dot(outputs.mean(), params=dict(net.named_parameters()))
