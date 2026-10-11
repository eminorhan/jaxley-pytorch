# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Port of Jaxley's `tests/test_optimize.py` (`optax` replaced by `torch.optim`)."""

import torch

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH
from jaxley_pytorch.optimize import TypeOptimizer
from jaxley_pytorch.optimize.utils import l2_norm
from tests.helpers import np


def _leaf_params(params):
    return [
        {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
        for p in params
    ]


def test_type_optimizer_api(SimpleComp):
    """Tests whether optimization recovers a ground truth parameter set."""
    comp = SimpleComp(copy=True)
    comp.insert(HH())
    comp.record()
    current = jx.step_current(
        i_delay=0.1, i_dur=3.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    comp.stimulate(current)

    def simulate(params):
        return jx.integrate(comp, params=params)

    comp.make_trainable("HH_gNa")
    comp.make_trainable("radius")

    def loss_fn(params):
        return torch.mean(simulate(params))

    # Diverse lr -> good learning.
    opt_params = _leaf_params(comp.get_parameters())
    lrs = {"HH_gNa": [0.01, 0.1], "radius": [1.0, 0.2]}
    optimizer = TypeOptimizer(
        lambda params, args: torch.optim.SGD(params, lr=args[0], momentum=args[1]),
        lrs,
        opt_params,
    )

    loss = loss_fn(opt_params)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_type_optimizer(SimpleComp):
    """Tests whether optimization recovers a ground truth parameter set."""
    comp = SimpleComp(copy=True)
    comp.insert(HH())
    comp.record()
    current = jx.step_current(
        i_delay=0.1, i_dur=3.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    comp.stimulate(current)

    comp.set("HH_gNa", 0.4)
    comp.set("radius", 30.0)

    def simulate(params):
        return jx.integrate(comp, params=params)

    observation = simulate([{}])
    comp.set("HH_gNa", 0.3)
    comp.make_trainable("HH_gNa")

    comp.set("radius", 20.0)
    comp.make_trainable("radius")

    def loss_fn(params):
        res = simulate(params)
        return torch.mean((observation - res) ** 2)

    def train(opt_params, optimizer, num_steps=500):
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(opt_params)
            loss.backward()
            optimizer.step()
        return loss.item()

    # Diverse lr -> good learning.
    opt_params = _leaf_params(comp.get_parameters())
    lrs = {"HH_gNa": 0.01, "radius": 1.0}
    optimizer = TypeOptimizer(
        lambda params, lr: torch.optim.Adam(params, lr=lr), lrs, opt_params
    )
    l = train(opt_params, optimizer)
    assert l < 1e-5, "Loss should be low if a diverse lr is used."

    # Too low lr -> poor learning.
    opt_params = _leaf_params(comp.get_parameters())
    optimizer = torch.optim.Adam(
        [v for p in opt_params for v in p.values()], lr=0.01
    )
    l = train(opt_params, optimizer)
    assert l > 3.0, "Loss should be high if a uniformly low lr is used."

    # Too high lr -> poor learning.
    opt_params = _leaf_params(comp.get_parameters())
    optimizer = torch.optim.Adam([v for p in opt_params for v in p.values()], lr=1.0)
    l = train(opt_params, optimizer)
    assert l > 30.0, "Loss should be high if a uniformly high lr is used."


def test_l2_norm_utility():
    true_norm = np.sqrt(np.sum(np.asarray([0.01, 0.003, 0.05, 0.006, 0.07, 0.04]) ** 2))
    pytree = [
        {"a": 0.01},
        {"b": np.asarray([[0.003, 0.05]])},
        {"c": np.asarray([[0.006, 0.07]])},
        0.04,
    ]
    assert l2_norm(pytree).item() == true_norm
