# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""The networks a run can train (`[model] name`): the rate network (`rate_network`) and
the networks of biophysical neurons (`spiking_network`), how a new one is drawn and how
a trained one is read."""

from typing import Callable, Dict, Tuple

import torch

from jaxley_pytorch.training import rate_network, spiking_network

MODELS = {"blk_rate": rate_network.BLKRate, **spiking_network.SPIKING_MODELS}


def _module(name: str):
    return rate_network if name == "blk_rate" else spiking_network


def new_network(
    model,
    init_io: Callable,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    compile: bool,
) -> Tuple[torch.nn.Module, Dict]:
    """The initial network of `model` (a `MODELS` instance) for `seed`, and its
    connectivity."""
    return _module(model.settings()["name"]).new_network(
        model, init_io, seed, device, dtype, compile
    )


def load_network(
    run_dir: str, device: torch.device, dtype: torch.dtype, compile: bool
) -> Tuple[torch.nn.Module, Dict, Dict]:
    """The trained network of a run directory, its connectivity and its settings."""
    name = rate_network.network_settings(run_dir)["model"]["name"]
    return _module(name).load_network(run_dir, device, dtype, compile)
