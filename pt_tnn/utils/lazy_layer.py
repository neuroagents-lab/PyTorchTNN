import torch
import torch.nn as nn
import torch.nn.init as init


# inherits from LazyLinear and override the reset_parameters() function
class LazyLinearWithInit(nn.LazyLinear):
    def __init__(self, out_features, init_dict=None, use_bias=True, bias_val=None):
        self.init_dict = init_dict
        self.use_bias = use_bias
        self.bias_val = bias_val
        super(LazyLinearWithInit, self).__init__(out_features, use_bias)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()
            # weight (re-)initialization
            if self.init_dict is not None:
                init_dict_copy = (
                    self.init_dict.copy()
                )  # Copy to avoid modifying the original dictionary
                init_method_name = init_dict_copy.pop(
                    "method"
                )  # Extract initialization method name
                init_method = getattr(
                    init, init_method_name
                )  # Get the initialization function from torch.nn.init
                init_method(
                    self.weight, **init_dict_copy
                )  # Initialize the weights using the specified method

            # Perform bias initialization
            if self.use_bias and self.bias_val is not None:
                init.constant_(self.bias, val=self.bias_val)


if __name__ == "__main__":
    # Instantiate the model with custom initialization
    model = LazyLinearWithInit(
        out_features=2,
        init_dict={"method": "constant_", "val": 0.3},
        use_bias=True,
        bias_val=0.2,
    )

    # Pass some input to trigger weight initialization
    dummy_input = torch.randn(5, 15)  # Batch of size 5, input features of size 15
    output = model(dummy_input)

    # Print the custom initialized weights and bias
    print("Custom initialized weights:", model.weight)
    print("Custom initialized bias:", model.bias)
