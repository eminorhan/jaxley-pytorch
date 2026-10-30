# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""`jaxley_pytorch.training.rate_network`: its simulation, and the same network simulated in
jaxley_pytorch, against the update of Beiran & Litwin-Kumar's code; the paper's
initialisation; the synapse layout."""

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import jaxley_pytorch as jx
from jaxley_pytorch.channels import Leak
from jaxley_pytorch.synapses import SoftplusRateSynapse
from jaxley_pytorch.training import rate_network as rn
from jaxley_pytorch.training.rate_network import E_L, S

SOLVER = "fwd_euler"
RADIUS, LENGTH = 10.0, 20.0  # um, point neurons
NA_PER_UNIT = 2 * math.pi * RADIUS * LENGTH * 1e-8 * 1e3  # nA per uA/cm^2


class JaxleySimulator(rn._Synapses):
    """`rn.Simulator`'s network as a jaxley_pytorch network, the reference the test
    compares it with: every neuron one passive compartment (leak 0.1 mS/cm^2, 1 uF/cm^2:
    tau = 10 ms) with membrane potential v_i = E_L + S * x_i, the input currents in
    uA/cm^2 (one model unit: 1 uA/cm^2 holds x_i at 1 against the leak), and
    current-based synapses (`SoftplusRateSynapse`) through which a neuron's gain and bias
    enter: gS_ij = J_ij * relu(g_j) (converted to nA), x_offset_ij = E_L - S * b_j and
    slope = 1 / S. Its forward-Euler step (`solver="fwd_euler"`) is the paper's update,
    but it evaluates the nonlinearity for every synapse (twice, for the slope of the
    current), over 100 times slower than `rn.Simulator` at 2,048 neurons."""

    def __init__(
        self,
        conn: Dict,
        device: torch.device,
        dtype: torch.dtype,
        dt: float,
        compile: bool = True,
    ):
        super().__init__(conn, device, dtype)
        self.dt, self.compile = dt, compile
        cell = jx.Cell()
        cell.set("radius", RADIUS)
        cell.set("length", LENGTH)
        cell.insert(Leak())
        cell.set("Leak_gLeak", 1e-4)
        cell.set("Leak_eLeak", E_L)
        net = jx.Network([cell] * self.n)
        nodes = net.nodes
        first = nodes[
            (nodes["local_branch_index"] == 0) & (nodes["local_comp_index"] == 0)
        ]
        first = first.sort_values("global_cell_index").index.to_numpy()
        assert len(first) == self.n
        pre, post = np.asarray(conn["pre"]), np.asarray(conn["post"])
        synapse = SoftplusRateSynapse()
        net._append_multiple_synapses(
            net.select(nodes=first[pre]).nodes,
            net.select(nodes=first[post]).nodes,
            synapse,
        )
        name = synapse._name
        net.edges[f"{name}_gS"] = 0.0
        net.edges[f"{name}_x_offset"] = E_L
        net.edges[f"{name}_slope"] = 1.0 / S
        net.set("v", E_L)
        net.init_states()
        for param in (f"{name}_gS", f"{name}_x_offset"):
            getattr(net, name).edge("all").make_trainable(param, verbose=False)
        net.cell("all").branch(0).comp(0).record("v", verbose=False)
        self.net, self.name = net, name
        self.input_view = net.select(nodes=first)
        self._template = {}

    def params(
        self, W: torch.Tensor, g: torch.Tensor, b: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """The trainable synapse parameters (in `make_trainable` order) for weights `W`
        (per synapse), gains `g` and biases `b` (per neuron)."""
        gs = self.weights(W) * torch.relu(g)[self.pre] * NA_PER_UNIT
        x_offset = E_L - S * b[self.pre]
        return [{f"{self.name}_gS": gs}, {f"{self.name}_x_offset": x_offset}]

    def rest_states(self, batch: int) -> Dict[str, torch.Tensor]:
        """Fresh states of `batch` networks with every membrane at rest (x = 0)."""
        if batch not in self._template:
            zeros = torch.zeros(batch, self.n, 1, device=self.device, dtype=self.dtype)
            none = torch.zeros_like(zeros[0, :, 0])
            dummy = self.params(self.real, none, none)
            with torch.no_grad():
                _, states = jx.integrate(
                    self.net,
                    params=dummy,
                    data_stimuli=self.input_view.data_stimulate(zeros),
                    delta_t=self.dt,
                    solver=SOLVER,
                    return_states=True,
                )
            self._template[batch] = {k: v.detach() for k, v in states.items()}
        return {
            k: torch.full_like(v, E_L) if k == "v" else v.clone()
            for k, v in self._template[batch].items()
        }

    def segment(
        self, states: Dict[str, torch.Tensor], current: torch.Tensor, params
    ) -> Tuple[torch.Tensor, Dict]:
        """Simulate input currents `current` (batch, n, steps; uA/cm^2) from `states`.
        Returns the voltages after every step (batch, n, steps) and the final states."""
        v, states = jx.integrate(
            self.net,
            params=params,
            data_stimuli=self.input_view.data_stimulate(current * NA_PER_UNIT),
            all_states=states,
            return_states=True,
            delta_t=self.dt,
            solver=SOLVER,
            compile=self.compile,
        )
        return v[..., 1:], states


def network(n, seed=0, dt=0.5, simulator=rn.Simulator):
    rng = np.random.default_rng(seed)
    g, b, mask, w, n_exc = rn.paper_init(n, rng, 0.7, 0.5)
    conn, w_edges = rn.edges_from_mask(mask, w, n_exc)
    sim = simulator(conn, torch.device("cpu"), torch.float64, dt, compile=False)
    return rn.RateNetwork(sim, w_edges, g, b, {}), conn


@pytest.mark.parametrize("simulator", [rn.Simulator, JaxleySimulator])
@pytest.mark.parametrize("dt", [0.5, 0.25])
def test_matches_the_papers_update(simulator, dt):
    """Rates, and the gradients of a random function of them, agree to rounding error
    with x <- x + alpha (-x + J a + I), alpha = dt / tau, over two segments (the second
    from the states the first returns), for the dense simulator and for the jaxley
    network."""
    n, batch, steps = 40, 3, 60
    net, conn = network(n, dt=dt, simulator=simulator)
    params = net.set_trainable(["W", "g", "b"])
    J = net.sim.dense_weights(net.W)
    dense = np.zeros((n, n))
    np.add.at(dense, (conn["post"], conn["pre"]), net.W.detach().numpy())
    assert torch.equal(J.detach(), torch.as_tensor(dense))
    gen = torch.Generator().manual_seed(0)
    current = 2.0 * torch.randn(batch, n, steps, generator=gen, dtype=torch.float64)
    probe = torch.randn(batch, n, steps, generator=gen, dtype=torch.float64)
    sim_params, states, vs = net.sim_params(), net.sim.rest_states(batch), []
    for piece in current.split(steps // 2, dim=-1):
        v, states = net.sim.segment(states, piece, sim_params)
        vs.append(v)
    a_sim = net.rates(torch.cat(vs, dim=-1))
    grads_sim = torch.autograd.grad((a_sim * probe).sum(), params)

    gain = torch.relu(net.g)
    x = torch.zeros(batch, n, dtype=torch.float64)
    rates = []
    for t in range(steps):
        a = gain * F.softplus(x + net.b)
        x = x + dt / rn.TAU * (-x + a @ J.T + current[..., t])
        rates.append(gain * F.softplus(x + net.b))
    a_ref = torch.stack(rates, dim=-1)
    grads_ref = torch.autograd.grad((a_ref * probe).sum(), params)

    def rel(a, b):
        return float(((a - b).abs().max() / b.abs().max()).detach())

    assert rel(a_sim, a_ref) < 1e-10
    for g_sim, g_ref in zip(grads_sim, grads_ref):
        assert rel(g_sim, g_ref) < 1e-10


def test_paper_init():
    """Dale's law, the connection probability, gains and biases."""
    g, b, mask, w, n_exc = rn.paper_init(200, np.random.default_rng(1), 0.7, 0.5)
    assert n_exc == 140
    assert np.all(w[:, :n_exc] >= 0) and np.all(w[:, n_exc:] <= 0)
    assert np.all(w[~mask] == 0)
    assert abs(mask.mean() - 0.5) < 0.02
    assert np.all(g > 0) and np.all(np.abs(b) <= 0.1)


def test_fan_in():
    """fan_in = n / 2 draws the paper's network; a smaller fan-in keeps a neuron's
    expected number of inputs."""
    half = rn.BLKRate(40, 0.7, dt=0.5, fan_in=20)
    paper = rn.paper_init(40, np.random.default_rng(3), 0.7, 0.5)
    drawn = rn.paper_init(40, np.random.default_rng(3), 0.7, half.p, half.weight_n)
    for a, b in zip(paper, drawn):
        np.testing.assert_array_equal(a, b)
    sparse = rn.BLKRate(200, 0.7, dt=0.5, fan_in=25)
    rng = np.random.default_rng(4)
    _, _, mask, _, _ = rn.paper_init(200, rng, 0.7, sparse.p, sparse.weight_n)
    assert sparse.p == 0.125 and sparse.weight_n == 50
    assert abs(mask.sum(1).mean() - 25) < 2


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
