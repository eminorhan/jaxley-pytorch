"""The rate network of Beiran & Litwin-Kumar (2025, Nat. Neurosci.) in jaxley_pytorch.

Every neuron is one passive compartment (leak 0.1 mS/cm^2, 1 uF/cm^2: time constant
tau = 10 ms) with membrane potential v_i. With x_i = (v_i - E_L) / S (S = 10 mV per model
unit), its output rate is

    a_i = relu(g_i) * softplus(x_i + b_i)

with a gain g_i and a bias b_i, and the forward-Euler step of jaxley_pytorch (dt = 0.5 ms,
`solver="fwd_euler"`) is exactly the update of the paper's code (alpha = dt / tau = 0.05):

    x_i <- x_i + alpha * (-x_i + sum_j J_ij a_j + I_i)

with input currents I_i in uA/cm^2 (one model unit: 1 uA/cm^2 holds x_i at 1 against the
leak). The synapses are current-based, `current = -gS * softplus((v_pre - x_offset) *
slope)` (`SoftplusRateSynapse`, written for this experiment); a neuron's gain and bias
enter through its outgoing synapses: gS_ij = J_ij * relu(g_j) (converted to nA),
x_offset_ij = E_L - S * b_j and slope = 1 / S. `train.py --role check` compares the
simulation with the update above.

Connectivity and initialisation are those of the paper's Fig. 1 teacher
(`nn_fig1_Teacher.py` in github.com/emebeiran/connconstr): 70% excitatory neurons, every
ordered pair connected with probability 0.5 (autapses included), weights
2.4 / sqrt(N) * normal + 1 / N, inhibitory ones scaled by 4, Dale's law as
`relu(W * sign) * sign` (a weight that crosses zero stays at zero); gains
~ normal(1, 0.4) (positive), biases ~ uniform(-0.1, 0.1). The synapses are listed per
postsynaptic neuron and padded with zero-weight synapses to the largest in-degree, the
layout in which jaxley_pytorch sums the synaptic currents without atomic adds
(`Network._step_regular_synapses`).
"""
import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import jaxley_pytorch as jx
from jaxley_pytorch.channels import Leak
from jaxley_pytorch.synapses import Synapse

DT = 0.5  # ms
TAU = 10.0  # ms
ALPHA = DT / TAU
E_L = -70.0  # mV (x = 0)
S = 10.0  # mV per unit of x
SOLVER = "fwd_euler"
RADIUS, LENGTH = 10.0, 20.0  # um, point neurons
NA_PER_UNIT = 2 * math.pi * RADIUS * LENGTH * 1e-8 * 1e3  # nA per uA/cm^2
STEPS_PER_TOKEN = 10  # 5 ms = tau / 2 per character or token
FRAME = 10  # solver steps per checkpointed piece of a cycling trial


class SoftplusRateSynapse(Synapse):
    """Current-based rate synapse, `current = -gS * softplus((v_pre - x_offset) * slope)`
    (nA; not part of Jaxley or jaxley_pytorch; no state). `gS > 0` depolarizes."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 0.0,
            f"{prefix}_x_offset": E_L,
            f"{prefix}_slope": 1.0 / S,
        }
        self.synapse_states = {}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        return {}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        rate = F.softplus((pre_voltage - params[f"{prefix}_x_offset"]) * params[f"{prefix}_slope"])
        return -params[f"{prefix}_gS"] * rate


def _softplus(x):
    return np.logaddexp(0.0, x)


def max_real_eigenvalue(w: np.ndarray, g: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    """The paper's stability test: largest real part of the eigenvalues of the linearised
    dynamics after 20 tau (step 0.1 tau) from a random state."""
    x = rng.standard_normal(len(g))
    for _ in range(200):
        x = x + 0.1 * (-x + w @ (g * _softplus(x + b)))
    slope = 0.5 * (1.0 + np.tanh(0.5 * (x + b)))  # sigmoid = softplus'
    return float(np.max(np.linalg.eigvals(w * (g * slope)[None, :]).real))


def random_gains_biases(n: int, rng: np.random.Generator):
    """The paper's initial gains and biases of a teacher: gains ~ normal(1, 0.4), redrawn
    until positive; biases ~ uniform(-0.1, 0.1)."""
    g = 0.4 * rng.standard_normal(n) + 1.0
    while np.any(g < 0):
        g[g < 0] = 0.4 * rng.standard_normal(int(np.sum(g < 0))) + 1.0
    b = 0.1 * (2 * rng.random(n) - 1.0)
    return g, b


def paper_init(n: int, rng: np.random.Generator, exc_frac: float = 0.7, p: float = 0.5):
    """Gains, biases, connection mask (post, pre) and weights of the paper's Fig. 1 teacher
    before training; connectivity redrawn until `max_real_eigenvalue` <= 1."""
    n_exc = int(exc_frac * n)
    g, b = random_gains_biases(n, rng)
    for _ in range(100):
        mask = rng.random((n, n)) < p
        w = (2.4 / np.sqrt(n) * rng.standard_normal((n, n)) + 1.0 / n) * mask
        w[:, n_exc:] *= 4.0
        w[:, :n_exc] = np.maximum(w[:, :n_exc], 0.0)
        w[:, n_exc:] = np.minimum(w[:, n_exc:], 0.0)
        if max_real_eigenvalue(w, g, b, rng) <= 1.0:
            return g, b, mask, w, n_exc
    raise RuntimeError("no stable connectivity in 100 draws")


def edges_from_mask(mask: np.ndarray, w: np.ndarray, n_exc: int) -> Tuple[Dict, np.ndarray]:
    """Synapses listed per postsynaptic neuron, padded with zero-weight synapses
    (`real == 0`) to the largest in-degree `k`. Returns the connectivity and the initial
    weight of every synapse."""
    n = mask.shape[0]
    k = int(mask.sum(1).max())
    pre = np.zeros((n, k), dtype=np.int64)
    real = np.zeros((n, k))
    w_edges = np.zeros((n, k))
    for i in range(n):
        js = np.nonzero(mask[i])[0]
        pre[i, : len(js)] = js
        pre[i, len(js) :] = js[0] if len(js) else i
        real[i, : len(js)] = 1.0
        w_edges[i, : len(js)] = w[i, js]
    conn = {"n": n, "n_exc": n_exc, "k": k, "pre": pre.ravel(), "post": np.repeat(np.arange(n), k),
            "real": real.ravel()}
    return conn, w_edges.ravel()


class Simulator:
    """The jaxley_pytorch network of a connectivity, shared by a teacher and its students
    (their parameters are passed to every simulation)."""

    def __init__(self, conn: Dict, device: torch.device, dtype: torch.dtype, compile: bool = True):
        self.n, self.n_exc, self.k = int(conn["n"]), int(conn["n_exc"]), int(conn["k"])
        self.device, self.dtype, self.compile = device, dtype, compile
        cell = jx.Cell()
        cell.set("radius", RADIUS)
        cell.set("length", LENGTH)
        cell.insert(Leak())
        cell.set("Leak_gLeak", 1e-4)
        cell.set("Leak_eLeak", E_L)
        net = jx.Network([cell] * self.n)
        nodes = net.nodes
        first = nodes[(nodes["local_branch_index"] == 0) & (nodes["local_comp_index"] == 0)]
        first = first.sort_values("global_cell_index").index.to_numpy()
        assert len(first) == self.n
        pre, post = np.asarray(conn["pre"]), np.asarray(conn["post"])
        synapse = SoftplusRateSynapse()
        net._append_multiple_synapses(net.select(nodes=first[pre]).nodes, net.select(nodes=first[post]).nodes, synapse)
        name = synapse._name
        net.edges[f"{name}_gS"] = 0.0
        net.edges[f"{name}_x_offset"] = E_L
        net.edges[f"{name}_slope"] = 1.0 / S
        net.set("v", E_L)
        net.init_states()
        getattr(net, name).edge("all").make_trainable(f"{name}_gS", verbose=False)
        getattr(net, name).edge("all").make_trainable(f"{name}_x_offset", verbose=False)
        net.cell("all").branch(0).comp(0).record("v", verbose=False)
        self.net, self.name = net, name
        self.input_view = net.select(nodes=first)
        as_t = lambda x, dt=dtype: torch.as_tensor(np.asarray(x), device=device, dtype=dt)
        self.pre = as_t(pre, torch.long)
        self.post = as_t(post, torch.long)
        self.sign = as_t(np.where(pre < self.n_exc, 1.0, -1.0))
        self.real = as_t(conn["real"])
        self._template = {}

    def edge_params(self, W: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """The trainable synapse parameters (in `make_trainable` order) for weights `W`
        (per synapse), gains `g` and biases `b` (per neuron)."""
        w = torch.relu(W * self.sign) * self.sign * self.real
        gs = w * torch.relu(g)[self.pre] * NA_PER_UNIT
        x_offset = E_L - S * b[self.pre]
        return [{f"{self.name}_gS": gs}, {f"{self.name}_x_offset": x_offset}]

    def dense_weights(self, W: torch.Tensor) -> torch.Tensor:
        """J (post, pre) after Dale's law and the mask."""
        w = torch.relu(W * self.sign) * self.sign * self.real
        J = torch.zeros(self.n, self.n, device=W.device, dtype=W.dtype)
        return J.index_put((self.post, self.pre), w, accumulate=True)

    def rest_states(self, batch: int) -> Dict[str, torch.Tensor]:
        """Fresh states of `batch` networks with every membrane at rest (x = 0)."""
        if batch not in self._template:
            zeros = torch.zeros(batch, self.n, 1, device=self.device, dtype=self.dtype)
            dummy = self.edge_params(self.real, torch.zeros_like(zeros[0, :, 0]), torch.zeros_like(zeros[0, :, 0]))
            with torch.no_grad():
                _, states = jx.integrate(self.net, params=dummy, data_stimuli=self.input_view.data_stimulate(zeros),
                                         delta_t=DT, solver=SOLVER, return_states=True)
            self._template[batch] = {k: v.detach() for k, v in states.items()}
        return {k: torch.full_like(v, E_L) if k == "v" else v.clone() for k, v in self._template[batch].items()}

    def segment(self, states: Dict[str, torch.Tensor], current: torch.Tensor, params) -> Tuple[torch.Tensor, Dict]:
        """Simulate input currents `current` (batch, n, steps; uA/cm^2) from `states`.
        Returns the voltages after every step (batch, n, steps) and the final states."""
        v, states = jx.integrate(
            self.net, params=params, data_stimuli=self.input_view.data_stimulate(current * NA_PER_UNIT),
            all_states=states, return_states=True, delta_t=DT, solver=SOLVER, compile=self.compile,
        )
        return v[..., 1:], states


class RateRNN(nn.Module):
    """One network (a teacher or a student): weights `W` (per synapse), gains `g` and
    biases `b` (per neuron), and the task's input and readout weights `io`. Only the
    parameters named in `trainable` require gradients.

    `io` of the cycling task: `w_in` (2, n) and `w_out` (n). Of the text tasks (width k):
    `embed` (vocab, k), `embed_proj` (n, k), `read_proj` (k, n), `unembed` (vocab, k) and
    `read_bias` (vocab): a token's input current is `embed_proj @ embed[token]`, the logits
    are `unembed @ (read_proj @ a) + read_bias`, with `a` the rates averaged over the
    token's window.
    """

    def __init__(self, sim: Simulator, W, g, b, io: Dict, trainable: Iterable[str]):
        super().__init__()
        self.sim = sim
        trainable = set(trainable)

        def param(name, value):
            t = torch.as_tensor(value).to(device=sim.device, dtype=sim.dtype).clone()
            return nn.Parameter(t, requires_grad=name in trainable)

        self.W, self.g, self.b = param("W", W), param("g", g), param("b", b)
        self.io = nn.ParameterDict({k: param(k, v) for k, v in io.items()})

    def rates(self, v: torch.Tensor) -> torch.Tensor:
        """Rates for voltages `v` (..., n, steps)."""
        return torch.relu(self.g)[:, None] * F.softplus((v - E_L) / S + self.b[:, None])

    def run_trials(self, inputs: torch.Tensor, states: Dict):
        """Cycling trials: `inputs` (batch, steps, 2) -> readout (batch, steps), rates
        (batch, n, steps) and the final states. As in the paper's code, step 0 reads the
        initial state and step t + 1 the state after the update with the input of step t."""
        params = self.sim.edge_params(self.W, self.g, self.b)
        current = torch.einsum("btc,cn->bnt", inputs, self.io["w_in"])
        steps = inputs.shape[1]
        rates = [self.rates(states["v"][..., None])]
        for f in range(0, steps, FRAME):
            piece = current[..., f : f + FRAME]
            if torch.is_grad_enabled():
                v, states = checkpoint(self.sim.segment, states, piece, params, use_reentrant=False)
            else:
                v, states = self.sim.segment(states, piece, params)
            rates.append(self.rates(v))
        a = torch.cat(rates, dim=-1)[..., :steps]
        return torch.einsum("bnt,n->bt", a, self.io["w_out"]), a, states

    def _token(self, states, current, params):
        stim = current[..., None].expand(*current.shape, STEPS_PER_TOKEN)
        v, states = self.sim.segment(states, stim, params)
        return self.rates(v).mean(-1), states

    def run_text(self, tokens: torch.Tensor, states: Dict):
        """`tokens` (batch, K) -> logits (batch, K, vocab), window-mean rates (batch, K, n)
        and the final states. Every token's simulation is checkpointed."""
        params = self.sim.edge_params(self.W, self.g, self.b)
        currents = self.io["embed"][tokens] @ self.io["embed_proj"].T
        feats = []
        for k in range(tokens.shape[1]):
            if torch.is_grad_enabled():
                a, states = checkpoint(self._token, states, currents[:, k], params, use_reentrant=False)
            else:
                a, states = self._token(states, currents[:, k], params)
            feats.append(a)
        feats = torch.stack(feats, dim=1)
        logits = (feats @ self.io["read_proj"].T) @ self.io["unembed"].T + self.io["read_bias"]
        return logits, feats, states

    def readout_basis(self) -> torch.Tensor:
        """Orthonormal basis (n, rank) of the activity directions the readout reads."""
        w = self.io["w_out"][:, None] if "w_out" in self.io else self.io["read_proj"].T
        q, _ = torch.linalg.qr(w.detach().double())
        return q
