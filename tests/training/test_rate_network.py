# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""`jaxley_pytorch.training.rate_network`: the simulation against the update of Beiran &
Litwin-Kumar's code, the paper's initialisation, and the synapse layout."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jaxley_pytorch.training import rate_network as rn


def network(n, seed=0, dt=0.5):
    rng = np.random.default_rng(seed)
    g, b, mask, w, n_exc = rn.paper_init(n, rng, 0.7, 0.5)
    conn, w_edges = rn.edges_from_mask(mask, w, n_exc)
    sim = rn.Simulator(conn, torch.device("cpu"), torch.float64, dt, compile=False)
    return rn.RateNetwork(sim, w_edges, g, b, {}), conn


@pytest.mark.parametrize("dt", [0.5, 0.25])
def test_matches_the_papers_update(dt):
    """Rates, and the gradients of a random function of them, agree to rounding error
    with x <- x + alpha (-x + J a + I), alpha = dt / tau."""
    n, batch, steps = 40, 3, 60
    net, _ = network(n, dt=dt)
    params = net.set_trainable(["W", "g", "b"])
    gen = torch.Generator().manual_seed(0)
    current = 2.0 * torch.randn(batch, n, steps, generator=gen)
    probe = torch.randn(batch, n, steps, generator=gen)
    v, _ = net.sim.segment(net.sim.rest_states(batch), current, net.edge_params())
    a_jx = net.rates(v)
    grads_jx = torch.autograd.grad((a_jx * probe).sum(), params)

    J = net.sim.dense_weights(net.W)
    gain = torch.relu(net.g)
    x = torch.zeros(batch, n)
    rates = []
    for t in range(steps):
        a = gain * F.softplus(x + net.b)
        x = x + dt / rn.TAU * (-x + a @ J.T + current[..., t])
        rates.append(gain * F.softplus(x + net.b))
    a_ref = torch.stack(rates, dim=-1)
    grads_ref = torch.autograd.grad((a_ref * probe).sum(), params)

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max())

    assert rel(a_jx, a_ref) < 1e-10
    for g_jx, g_ref in zip(grads_jx, grads_ref):
        assert rel(g_jx, g_ref) < 1e-10


def test_paper_init():
    """Dale's law, the connection probability, gains and biases."""
    g, b, mask, w, n_exc = rn.paper_init(200, np.random.default_rng(1), 0.7, 0.5)
    assert n_exc == 140
    assert np.all(w[:, :n_exc] >= 0) and np.all(w[:, n_exc:] <= 0)
    assert np.all(w[~mask] == 0)
    assert abs(mask.mean() - 0.5) < 0.02
    assert np.all(g > 0) and np.all(np.abs(b) <= 0.1)


def test_edges_from_mask():
    """Every synapse once, listed per postsynaptic neuron, padded with zero weights."""
    rng = np.random.default_rng(0)
    mask = rng.random((7, 7)) < 0.4
    w = rng.standard_normal((7, 7)) * mask
    conn, w_edges = rn.edges_from_mask(mask, w, 4)
    assert conn["k"] == mask.sum(1).max()
    real = conn["real"].astype(bool)
    assert real.sum() == mask.sum()
    dense = np.zeros((7, 7))
    np.add.at(dense, (conn["post"][real], conn["pre"][real]), w_edges[real])
    np.testing.assert_array_equal(dense, w)
    assert np.all(w_edges[~real] == 0)
    assert np.all(np.diff(conn["post"]) >= 0)
