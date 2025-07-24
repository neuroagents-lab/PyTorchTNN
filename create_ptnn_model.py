from pt_tnn.temporal_graph import TemporalGraph
import torch

model_config_file = (
    "configs/test_resnet18_bn_timevary.json"  # specify the json file configuration
)
input_shape = [3, 6, 6]  # specify the input shape (C, H, W)
num_timesteps = 10  # number of unroll times

## creating the model
TG = TemporalGraph(
    model_config_file=model_config_file,
    # recurrent_module=YourCustomizedRecurrentModule,
    # (default: None, which means using the RecurrentModule from pt_tnn)
    input_shape=input_shape,
    num_timesteps=num_timesteps,
    transform=None,
)

bs = 5
random_input = torch.rand(
    (
        bs,
        num_timesteps,
    )
    + tuple(input_shape)
)
# (bs, T, C, H, W) or (bs, C, H, W) where the same input will be repeated T times


## forward pass
output, activations = TG(
    random_input,
    n_times=num_timesteps,  # how many times to unroll
    return_all=True,
    # whether return all the time steps,
    return_activations=True,
    # for neural fitting (True, a dictionary of {t:{layer_name:tensor}}),
    # default is False (and `activations` will not be returned)
)
# pytorch implementation of: https://github.com/neuroailab/convrnns/blob/master/convrnns/utils/main.py#L190

full_unroll_output, full_unroll_activations = TG.full_unroll(
    random_input,
    n_times=num_timesteps,  # how many times to unroll
    return_all=True,
    # whether return all the time steps,
    return_activations=True,
    # for neural fitting (True, a dictionary of {t:{layer_name:tensor}}),
    # default is False (and `full_unroll_activations` will not be returned)
)
# pytorch implementation of: https://github.com/neuroailab/convrnns/blob/master/convrnns/utils/main.py#L323


print("shape of output:", output.shape)  # (bs, T, num_class)
print("all time steps in the activations:", activations.keys())
print(
    "different layers at some time step (e.g., t=0) in the activations:",
    activations[0].keys(),
)
print(
    "shape of activations at time=0, layer=conv1", activations[0]["conv1"].shape
)  # (bs, C', H', W')

# results are identical for full_unroll_output, full_unroll_activations
