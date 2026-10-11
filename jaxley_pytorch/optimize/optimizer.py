# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


class TypeOptimizer:
    """Optimizer wrapper which allows different argument values for different params.

    This is the PyTorch counterpart of Jaxley's `optax` wrapper. Every entry of
    `opt_params` (one dict per call of `make_trainable`) gets its own optimizer, built
    with the arguments of its parameter name.
    """

    def __init__(
        self,
        optimizer: Callable[[list[Tensor], Any], torch.optim.Optimizer],
        optimizer_args: dict[str, Any],
        opt_params: list[dict[str, Tensor]],
    ):
        """Create the optimizers.

        This requires access to `opt_params` in order to know how many optimizers
        should be created. It creates `len(opt_params)` optimizers. The tensors in
        `opt_params` are the leaf tensors that are optimized (they must have
        `requires_grad=True`).

        Example usage:
        ```
        lrs = {"HH_gNa": 0.01, "radius": 1.0}
        optimizer = TypeOptimizer(
            lambda params, lr: torch.optim.Adam(params, lr=lr), lrs, opt_params
        )
        loss = loss_fn(opt_params)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ```

        ```
        optimizer_args = {"HH_gNa": [0.01, 0.4], "radius": [1.0, 0.8]}
        optimizer = TypeOptimizer(
            lambda params, args: torch.optim.SGD(params, lr=args[0], momentum=args[1]),
            optimizer_args,
            opt_params
        )
        ```

        Args:
            optimizer: A Callable that takes the list of tensors and the arguments
                and returns the `torch.optim.Optimizer` which should be used.
            optimizer_args: The arguments for different kinds of parameters.
                Each item of the dictionary will be passed to the `Callable` passed to
                `optimizer`.
            opt_params: The parameters to be optimized.
        """
        self.base_optimizer = optimizer

        self.optimizers = []
        for params in opt_params:
            names = list(params.keys())
            assert len(names) == 1, "Multiple parameters were added at once."
            name = names[0]
            optimizer = self.base_optimizer(list(params.values()), optimizer_args[name])
            self.optimizers.append({name: optimizer})

    def step(self):
        """Take one step with every optimizer."""
        for opt in self.optimizers:
            for optimizer in opt.values():
                optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        """Reset the gradients of all parameters."""
        for opt in self.optimizers:
            for optimizer in opt.values():
                optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> list:
        return [
            {name: optimizer.state_dict() for name, optimizer in opt.items()}
            for opt in self.optimizers
        ]

    def load_state_dict(self, state_dicts: list):
        for opt, state in zip(self.optimizers, state_dicts):
            for name, optimizer in opt.items():
                optimizer.load_state_dict(state[name])
