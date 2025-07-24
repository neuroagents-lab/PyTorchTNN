import json
import copy
import itertools
import networkx as nx
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List
from . import recurrent_module as rm

__all__ = ["TemporalGraph"]


class TemporalGraph(nn.Module):
    """
    Definition of the computation graph, including reading the graph from the
    configuration file and unrolling the graph in time.
    """

    def __init__(
        self,
        model_config_file,
        recurrent_module: nn.Module = None,
        input_shape: List = None,
        num_timesteps: int = None,
        transform: Callable = None,
    ):
        super(TemporalGraph, self).__init__()

        self._model_config_file = model_config_file
        self._config_info = self._read_config(
            input_shape=input_shape, num_timesteps=num_timesteps
        )
        self._G = self._setup_graph_attrs()
        self.transform = nn.Identity() if transform is None else transform

        with torch.no_grad():
            self._determine_recurrent_module_shapes(recurrent_module=recurrent_module)

        self.topo_sorted_nodes = self._topological_sort(ff_order=self.ff_order)

    @property
    def input_shape(self):
        return tuple(self._config_info["input_shape"])

    @property
    def nodes(self):
        return self._config_info["nodes"]

    @property
    def edges(self):
        return self._config_info["edges"]

    @property
    def input_nodes(self):
        return self._config_info["input_nodes"]

    @property
    def output_node(self):
        return self._config_info["output_node"]

    @property
    def node_names(self):
        return self._config_info["node_names"]

    @property
    def num_timesteps(self):
        return self._config_info["num_timesteps"]

    @property
    def ff_order(self):
        return self._config_info["ff_order"]

    @property
    def graph(self):
        return self._G

    def cuda(self):
        for node, attr in self.graph.nodes(data=True):
            _mod = getattr(self, node).cuda()
            setattr(self, node, _mod)
        return self

    def _check_nodes_edges(self, nodes, edges):
        node_names = [n["name"] for n in nodes]

        assert len(edges) > 0
        assert set(itertools.chain(*edges)) == set(
            node_names
        ), "Nodes and edges do not match."

        return node_names

    def _read_config(self, input_shape, num_timesteps):
        # Read the graph configuration file
        if isinstance(self._model_config_file, dict):
            config = (
                self._model_config_file
            )  # already loaded json dict (can be modified outside)
        else:
            with open(self._model_config_file, "r") as f:
                config = json.load(f)

        # Grab and check nodes and edges for the graph
        nodes = config["nodes"]
        edges = [(str(i["from"]), str(i["to"])) for i in config["edges"]]
        node_names = self._check_nodes_edges(nodes, edges)

        # Grab input node
        input_nodes = config["input_nodes"]
        for in_node in input_nodes:
            assert in_node in node_names

        # Grab output node
        output_node = config["output_node"]
        assert output_node in node_names

        # Grab input shape
        input_shape = config["input_shape"] if input_shape is None else input_shape
        # override the config if input_shape is provided
        assert (
            len(input_shape) == 3
        ), "`input_shape' must be of length 3 of format (C, H, W)"

        # Grab number of times to unroll. This argument is used to specify whether one
        # wants to have timestep-specific batch norm. If not specified, the same batch
        # norm is used for all timesteps.
        num_timesteps = (
            config.get("num_timesteps", None)
            if num_timesteps is None
            else num_timesteps
        )
        # override the config if num_timesteps is provided
        ff_order = config.get("ff_order", None)

        output_dict = {
            "input_shape": input_shape,
            "nodes": nodes,
            "edges": edges,
            "input_nodes": input_nodes,
            "output_node": output_node,
            "node_names": node_names,
            "num_timesteps": num_timesteps,
            "ff_order": ff_order,  # added here, to manually specify the order
        }

        return output_dict

    def _setup_graph_attrs(self):
        # Create graph
        G = nx.DiGraph(self.edges)

        # Loop through each node and set the attributes
        for n in self.nodes:
            attr = G.nodes[n["name"]]
            attr["name"] = n["name"]

            # Tells us where the spatial shape of the input for this recurrent module
            # comes from
            attr["shape_from"] = n["shape_from"]

            # Tells us the spatial shape of the output for this recurrent module
            attr["output_shape"] = None

            # Tells us the number of output channels for this recurrent module
            attr["out_channels"] = n["out_channels"]

            # Determine the number of input channels.
            attr["in_channels"] = list()
            attr["ff_channels"] = (
                None  # we only assume there's one feedforward connection
            )
            for _pred in G.predecessors(n["name"]):
                _idx = self.node_names.index(_pred)
                _out_channels = self.nodes[_idx]["out_channels"]
                attr["in_channels"].append(_out_channels)
                if _pred == attr["shape_from"]:
                    attr["ff_channels"] = _out_channels

            # Input nodes will also "receive" channels from the input data.
            if n["name"] in self.input_nodes:
                attr["in_channels"].append(self.input_shape[0])
                attr["ff_channels"] = self.input_shape[0]

            attr["residual"] = n.get("residual", None)

        return G

    def _get_module_output_shape(self, node, attr):
        # Forward pass through the recurrent module
        output, _ = getattr(self, node)(curr_timestep=0)  # dummy timestep

        # Checks for number of output channels in the current recurrent module
        assert output.ndim == 4 or output.ndim == 2  # support fc layer
        assert output.shape[1] == attr["out_channels"], (
            f"Expected {attr['out_channels']} channels, but "
            + f"got {output.shape[1]} channels."
        )

        return output

    def _update_harbor_input_shapes_and_initialize(self):
        """
        This function obtains the correct input shapes for the recurrent module
        and initializes the harbor policy operations.
        """
        for node, attr in self.graph.nodes(data=True):
            curr_harbor_policy = getattr(self, node).harbor_policy
            output_shapes = list()
            _channels = list()
            for pred in sorted(self.graph.predecessors(node)):
                output_shape = self.graph.nodes[pred][
                    "output_shape"
                ]  # a tuple with 2 or 0 element
                out_channels = self.graph.nodes[pred]["out_channels"]
                assert (
                    len(output_shape) == 2 or len(output_shape) == 0
                )  # add support for fc layer
                pred_output_shape = (
                    out_channels,
                ) + output_shape  # (C, H, W) or (num_class, )
                output_shapes.append(pred_output_shape)
                _channels.append(out_channels)

            if node in self.input_nodes:
                output_shapes.append(self.input_shape)
                _channels.append(self.input_shape[0])

            assert set(_channels) == set(attr["in_channels"])
            curr_harbor_policy.initialize_operations(output_shapes)

    def _determine_recurrent_module_shapes(self, recurrent_module):
        """
        This function is used to determine the output sizes of each recurrent
        module given a desired input shape of the ConvRNN model.
        """
        # Set up record of visited nodes
        visited = list()

        # Set up node-exploration queue
        nodes_to_explore = copy.deepcopy(self.input_nodes)

        # Breadth-first traversal of the graph
        while len(nodes_to_explore) > 0:
            curr_node = nodes_to_explore[0]
            nodes_to_explore.pop(0)

            # Dictionary for node attributes
            attr = self.graph.nodes[curr_node]

            # Set the recurrent module's input spatial size
            if attr["shape_from"] is None:
                assert curr_node in self.input_nodes
                spatial_shape = (
                    self.input_shape[1],
                    self.input_shape[2],
                )  # line 99: self.input_shape (C, H, W)
            else:
                spatial_shape = self.graph.nodes[attr["shape_from"]]["output_shape"]
                # "shape_from" gets the predecessor of current node, "output_shape" is None by default,
                # "output_shape" will be (H, W)

            # If spatial_shape is None, it means that the node's direct parent has not
            # been explored yet. So we postpone exploring the current node
            if spatial_shape is None:
                nodes_to_explore.append(curr_node)
                continue
            attr["input_shape"] = spatial_shape

            # Instantiate the recurrent module
            node_idx = self.node_names.index(curr_node)
            if recurrent_module is None:
                _mod = rm.RecurrentModule(
                    self.nodes[node_idx]["harbor_policy"],
                    self.nodes[node_idx]["pre_memory"],
                    self.nodes[node_idx]["recurrent_cell"],
                    self.nodes[node_idx]["post_memory"],
                    attr,
                    self.num_timesteps,
                )
            else:
                _mod = recurrent_module(  # use customized recurrent module
                    self.nodes[node_idx]["harbor_policy"],
                    self.nodes[node_idx]["pre_memory"],
                    self.nodes[node_idx]["recurrent_cell"],
                    self.nodes[node_idx]["post_memory"],
                    attr,
                    self.num_timesteps,
                )
            setattr(self, curr_node, _mod)

            # Get module's output shape
            output = self._get_module_output_shape(curr_node, attr)

            # Set the spatial shape of the output for the recurrent module
            attr["output_shape"] = tuple(output.shape[2:])
            # update the output shape with real data, which will be later used

            # Update list of input shapes for each successor node
            assert (
                output.ndim == 4 or output.ndim == 2
            )  # (N, C, H, W) or (N, num_class) [support fc layer]
            for next_node in self.graph.successors(curr_node):
                next_node_attr = self.graph.nodes[next_node]

                # If next node has not been visited and is not already in the queue,
                # then add it to the queue
                if next_node not in visited and next_node not in nodes_to_explore:
                    nodes_to_explore.append(next_node)

            # Update visited data structure
            visited.append(curr_node)

        assert len(visited) == self.graph.number_of_nodes()

        # Finally, update all the input shapes
        self._update_harbor_input_shapes_and_initialize()

    def _topological_sort(self, ff_order):

        def check_inputs(G, input_nodes):
            """Given a networkx graph G and a set of input_nodes,
            checks whether the inputs are valid"""

            for n in input_nodes:
                if n not in G.nodes():
                    raise ValueError(
                        "The input nodes provided must all be in the graph."
                    )

            input_cover = set([])
            for n in input_nodes:
                input_cover |= set([n]) | set(nx.descendants(G, n))

            if input_cover != set(G.nodes()):
                missed_nodes = ", ".join(list(set(G.nodes()) - input_cover))
                raise ValueError(
                    "Not all valid input nodes have been provided, as the following nodes will not receive any data: {}".format(
                        missed_nodes
                    )
                )

        # find the longest path from the inputs to the outputs:
        check_inputs(self.graph, self.input_nodes)
        output_nodes = (
            [self.output_node]
            if isinstance(self.output_node, str)
            else self.output_node
        )

        inp_out = itertools.product(self.input_nodes, output_nodes)
        paths = []
        for inp, out in inp_out:
            paths.extend([p for p in nx.all_simple_paths(self.graph, inp, out)])

        path_lengths = map(len, paths)
        longest_path_len = max(path_lengths) if path_lengths else 0

        try:
            # sort nodes in topological order (very efficient)
            # will only work for directed graphs (so no feedbacks), otherwise always correct ordering
            s = [n for n in nx.topological_sort(self.graph)]
        except:
            # in the event there are feedbacks
            # go with the union of the simple paths (not as efficient) between multiple inputs/outputs

            if ff_order is not None:
                assert isinstance(ff_order, list)
                s = ff_order
            else:
                # find a longest simple path
                longest_max_p = None
                for p in paths:
                    if len(p) == longest_path_len:
                        longest_max_p = p
                        break

                s = longest_max_p  # likely will contain most of the nodes already
                for p in paths:
                    for n in p:
                        if n not in s:
                            is_pred = False
                            is_succ = False
                            for idx, existing_n in enumerate(s):
                                # find first node that n is a predecessor of
                                if n in self.graph.predecessors(existing_n):
                                    s.insert(idx, n)
                                    is_pred = True
                                    break
                            # if n is not a predecessor of anything currently in s, it is a separate output node
                            if not is_pred:
                                # find last node that n is a successor of
                                successor_idxs = []
                                for idx, existing_n in enumerate(s):
                                    if n in self.graph.successors(existing_n):
                                        successor_idxs.append(idx)

                                if len(successor_idxs) > 0:
                                    s.insert(successor_idxs[-1] + 1, n)
                                    is_succ = True

                            # n is neither a predecessor or successor
                            # then n must be the input node of a separate path, insert at the beginning
                            if not is_pred and not is_succ:
                                s.insert(0, n)

                print("Cannot topologically sort, assuming this ordering: ", s)
                print(
                    "If you do not want this ordering, pass your own ordering via ff_order"
                )

        # assert all nodes in ordering
        assert set(s) == set(self.graph.nodes())
        return s

    def forward(self, inputs, n_times=None, return_all=False, return_activations=False):
        """
        This is the main function call for a forward pass of the model in time

        Args:
            inputs (torch.Tensor): batch of image inputs (N, T, C, H, W) or
                (N, C, H, W). (batch, time, channel, height, width). If 5D input is
                provided, then n_times must be the same as the time dimension of the
                input. Otherwise, the same input batch is used for all time steps.
            n_times (int): number of time steps to unroll the graph. Default: None.
        """
        assert n_times is not None
        if inputs.ndim == 5:
            assert (
                inputs.shape[1] == n_times
            ), "Time length of inputs should be the same as the number of unroll steps."
            assert tuple(inputs.shape[2:]) == self.input_shape
        else:
            assert inputs.ndim == 4
            assert tuple(inputs.shape[1:]) == self.input_shape

        inputs = self.transform(
            inputs
        )  # added transformation on data here (default is nn.Identity)

        outputs = dict()
        states = dict()
        last_layer_outputs = []
        activations = dict()

        for t in range(n_times):
            if inputs.ndim == 5:
                curr_input = inputs[:, t, :, :, :]
            else:
                curr_input = inputs

            new_outputs = dict()
            new_states = dict()
            for node, attr in self.graph.nodes(data=True):
                curr_node_inputs = list()

                if node in self.input_nodes:
                    curr_node_inputs.append(curr_input)

                for pred in sorted(self.graph.predecessors(node)):
                    if t == 0:
                        _zeros = torch.zeros(
                            inputs.shape[0],
                            self.graph.nodes[pred]["out_channels"],
                            *self.graph.nodes[pred]["output_shape"],
                            device=curr_input.device,
                        )
                        curr_node_inputs.append(_zeros)
                        # init zero-input for non-input nodes
                    else:
                        curr_node_inputs.append(outputs[pred])

                state = None if t == 0 else states[node]

                dummy_input_type = ["ff_input"] + ["fb_input"] * (
                    len(curr_node_inputs) - 1
                )  # note that the feedback connection is defined in the config
                output, state = getattr(self, node)(
                    inputs=curr_node_inputs,
                    input_types=dummy_input_type,
                    state=state,
                    curr_timestep=t,
                )
                new_outputs[node] = output
                new_states[node] = state

            outputs = new_outputs
            states = new_states
            last_layer_outputs.append(outputs[self.output_node])
            activations[t] = outputs

        if return_all:
            if return_activations:
                return torch.stack(last_layer_outputs, dim=1), activations
            else:
                return torch.stack(last_layer_outputs, dim=1)  # (bs, T, d)
        else:
            if return_activations:
                return last_layer_outputs[-1], activations
            else:
                return last_layer_outputs[-1]
            # last time step, (bs, d)

    def full_unroll(
        self, inputs, n_times=None, return_all=False, return_activations=False
    ):
        """
        This is the recurrence call for a forward pass of the model

        Args:
            inputs (torch.Tensor): batch of image inputs (N, T, C, H, W) or
                (N, C, H, W). (batch, time, channel, height, width). If 5D input is
                provided, then n_times must be the same as the time dimension of the
                input. Otherwise, the same input batch is used for all time steps.
            n_times (int): number of time steps to unroll the graph. Default: None.
        """
        assert n_times is not None
        if inputs.ndim == 5:
            assert (
                inputs.shape[1] == n_times
            ), "Time length of inputs should be the same as the number of unroll steps."
            assert tuple(inputs.shape[2:]) == self.input_shape
        else:
            assert inputs.ndim == 4
            assert tuple(inputs.shape[1:]) == self.input_shape

        inputs = self.transform(
            inputs
        )  # added transformation on data here (default is nn.Identity)

        outputs = dict()
        states = dict()
        last_layer_outputs = []
        activations = dict()

        for t in range(n_times):
            if inputs.ndim == 5:
                curr_input = inputs[:, t, :, :, :]
            else:
                curr_input = inputs

            new_outputs = dict()
            # record outputs for all the nodes, no need to access the attributes of nodes in self.graph
            new_states = dict()
            for node in self.topo_sorted_nodes:
                # attr = self.graph.nodes[node]
                curr_node_inputs = list()

                if node in self.input_nodes:
                    curr_node_inputs.append(curr_input)

                for pred in sorted(self.graph.predecessors(node)):
                    pred_idx = self.topo_sorted_nodes.index(pred)
                    curr_idx = self.topo_sorted_nodes.index(node)

                    if (
                        curr_idx > pred_idx
                    ):  # feedforward / skip connection (always have outputs from previous layers)
                        curr_node_inputs.append(
                            new_outputs[pred]
                        )  # `new_outputs`: outputs at time t (current step)
                    else:  # feedback connection
                        if t == 0:
                            _zeros = torch.zeros(
                                inputs.shape[0],
                                self.graph.nodes[pred]["out_channels"],
                                *self.graph.nodes[pred]["output_shape"],
                                device=curr_input.device,
                            )
                            curr_node_inputs.append(_zeros)
                        else:
                            curr_node_inputs.append(outputs[pred])
                            # `outputs`: outputs at time t-1 (previous step)

                state = None if t == 0 else states[node]

                dummy_input_type = ["ff_input"] + ["fb_input"] * (
                    len(curr_node_inputs) - 1
                )  # note that the feedback connection is defined in the config
                output, state = getattr(self, node)(
                    inputs=curr_node_inputs,
                    input_types=dummy_input_type,
                    state=state,
                    curr_timestep=t,
                )
                new_outputs[node] = output
                new_states[node] = state

            outputs = new_outputs
            states = new_states
            last_layer_outputs.append(outputs[self.output_node])
            activations[t] = outputs

        if return_all:
            if return_activations:
                return torch.stack(last_layer_outputs, dim=1), activations
            else:
                return torch.stack(last_layer_outputs, dim=1)  # (bs, T, d)
        else:
            if return_activations:
                return last_layer_outputs[-1], activations
            else:
                return last_layer_outputs[-1]
            # last time step, (bs, d)
