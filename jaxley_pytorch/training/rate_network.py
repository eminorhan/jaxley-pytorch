# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""The rate network of Beiran & Litwin-Kumar (2025, Nat. Neurosci. 28:2561), selected by
`[model] name = "blk_rate"`.

Every neuron is one passive compartment (leak 0.1 mS/cm^2, 1 uF/cm^2: time constant
tau = 10 ms) with membrane potential v_i. With x_i = (v_i - E_L) / S (S = 10 mV per
model unit), its output rate is

    a_i = relu(g_i) * softplus(x_i + b_i)

with a gain g_i and a bias b_i, and the forward-Euler step of jaxley_pytorch (time step
dt, `solver="fwd_euler"`) is exactly the update of the paper's code (alpha = dt / tau):

    x_i <- x_i + alpha * (-x_i + sum_j J_ij a_j + I_i)

with input currents I_i in uA/cm^2 (one model unit: 1 uA/cm^2 holds x_i at 1 against the
leak). The synapses are current-based (`jaxley_pytorch.synapses.SoftplusRateSynapse`); a
neuron's gain and bias enter through its outgoing synapses: gS_ij = J_ij * relu(g_j)
(converted to nA), x_offset_ij = E_L - S * b_j and slope = 1 / S.
`tests/training/test_rate_network.py` checks the simulation against the update above.

Connectivity and initialisation are those of the paper's Fig. 1 teacher
(`nn_fig1_Teacher.py` in github.com/emebeiran/connconstr): a fraction `exc_frac` of the
neurons excitatory, every ordered pair connected with probability `connection_prob`
(autapses included), weights 2.4 / sqrt(N) * normal + 1 / N, inhibitory ones scaled by
4, Dale's law as `relu(W * sign) * sign` (a weight that crosses zero stays at zero);
gains ~ normal(1, 0.4) (positive), biases ~ uniform(-0.1, 0.1); the connectivity is
drawn again until the paper's stability test passes. The synapses are listed per
postsynaptic neuron and padded with zero-weight synapses to the largest in-degree, the
layout in which jaxley_pytorch sums the synaptic currents without atomic adds
(`Network._step_regular_synapses`).
"""

import math
import os
import time
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import jaxley_pytorch as jx
from jaxley_pytorch.channels import Leak
from jaxley_pytorch.synapses import SoftplusRateSynapse

TAU = 10.0  # ms
E_L = -70.0  # mV (x = 0)
S = 10.0  # mV per unit of x
SOLVER = "fwd_euler"
RADIUS, LENGTH = 10.0, 20.0  # um, point neurons
NA_PER_UNIT = 2 * math.pi * RADIUS * LENGTH * 1e-8 * 1e3  # nA per uA/cm^2


class BLKRate:
    """`[model] name = "blk_rate"`: `neurons` rate neurons, a fraction `exc_frac` of
    them excitatory, every ordered pair connected with probability `connection_prob`,
    simulated with a time step of `dt` ms."""

    def __init__(
        self, neurons: int, exc_frac: float, connection_prob: float, dt: float
    ):
        if not (isinstance(neurons, int) and neurons >= 2):
            raise ValueError(
                f"[model] neurons must be an integer >= 2, got {neurons!r}"
            )
        if not 0.0 < exc_frac < 1.0:
            raise ValueError(f"[model] exc_frac must be in (0, 1), got {exc_frac!r}")
        if not 0.0 < connection_prob <= 1.0:
            raise ValueError(
                f"[model] connection_prob must be in (0, 1], got {connection_prob!r}"
            )
        if not dt > 0.0:
            raise ValueError(f"[model] dt must be positive, got {dt!r}")
        self.neurons, self.exc_frac = neurons, exc_frac
        self.connection_prob, self.dt = connection_prob, dt

    def settings(self) -> Dict:
        return {
            "name": "blk_rate",
            "neurons": self.neurons,
            "exc_frac": self.exc_frac,
            "connection_prob": self.connection_prob,
            "dt": self.dt,
        }


def _softplus(x):
    return np.logaddexp(0.0, x)


def max_real_eigenvalue(
    w: np.ndarray, g: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> float:
    """The paper's stability test: largest real part of the eigenvalues of the
    linearised dynamics after 20 tau (step 0.1 tau) from a random state."""
    x = rng.standard_normal(len(g))
    for _ in range(200):
        x = x + 0.1 * (-x + w @ (g * _softplus(x + b)))
    slope = 0.5 * (1.0 + np.tanh(0.5 * (x + b)))  # sigmoid = softplus'
    return float(np.max(np.linalg.eigvals(w * (g * slope)[None, :]).real))


def random_gains_biases(n: int, rng: np.random.Generator):
    """The paper's initial gains and biases of a teacher: gains ~ normal(1, 0.4),
    redrawn until positive; biases ~ uniform(-0.1, 0.1)."""
    g = 0.4 * rng.standard_normal(n) + 1.0
    while np.any(g < 0):
        g[g < 0] = 0.4 * rng.standard_normal(int(np.sum(g < 0))) + 1.0
    b = 0.1 * (2 * rng.random(n) - 1.0)
    return g, b


def paper_init(n: int, rng: np.random.Generator, exc_frac: float, p: float):
    """Gains, biases, connection mask (post, pre), weights and number of excitatory
    neurons of the paper's Fig. 1 teacher before training; the connectivity is drawn
    again until `max_real_eigenvalue` <= 1."""
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


def edges_from_mask(
    mask: np.ndarray, w: np.ndarray, n_exc: int
) -> Tuple[Dict, np.ndarray]:
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
    conn = {
        "n": n,
        "n_exc": n_exc,
        "k": k,
        "pre": pre.ravel(),
        "post": np.repeat(np.arange(n), k),
        "real": real.ravel(),
    }
    return conn, w_edges.ravel()


class Simulator:
    """The jaxley_pytorch network of a connectivity, shared by a teacher and its
    students (their parameters are passed to every simulation)."""

    def __init__(
        self,
        conn: Dict,
        device: torch.device,
        dtype: torch.dtype,
        dt: float,
        compile: bool = True,
    ):
        self.n, self.n_exc, self.k = int(conn["n"]), int(conn["n_exc"]), int(conn["k"])
        self.device, self.dtype, self.dt, self.compile = device, dtype, dt, compile
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

        def as_t(x, dt=dtype):
            return torch.as_tensor(np.asarray(x), device=device, dtype=dt)

        self.pre = as_t(pre, torch.long)
        self.post = as_t(post, torch.long)
        self.sign = as_t(np.where(pre < self.n_exc, 1.0, -1.0))
        self.real = as_t(conn["real"])
        self._template = {}

    def edge_params(
        self, W: torch.Tensor, g: torch.Tensor, b: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
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
            none = torch.zeros_like(zeros[0, :, 0])
            dummy = self.edge_params(self.real, none, none)
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


class RateNetwork(nn.Module):
    """One network: weights `W` (per synapse), gains `g` and biases `b` (per neuron),
    and a task's input and readout weights `io` (see `tasks.py`), simulated by `sim`,
    which a teacher shares with its students. No parameter requires gradients until
    `set_trainable`."""

    def __init__(self, sim: Simulator, W, g, b, io: Dict):
        super().__init__()
        self.sim = sim
        self.W, self.g, self.b = (self._param(x) for x in (W, g, b))
        self.io = nn.ParameterDict({k: self._param(v) for k, v in io.items()})

    def _param(self, value) -> nn.Parameter:
        t = torch.as_tensor(value).to(device=self.sim.device, dtype=self.sim.dtype)
        return nn.Parameter(t.clone(), requires_grad=False)

    def named_weights(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """(name, parameter) for "W", "g", "b" and the task's input and readout
        weights, in this order."""
        yield from (("W", self.W), ("g", self.g), ("b", self.b))
        yield from self.io.items()

    def set_trainable(self, names: Iterable[str]) -> List[nn.Parameter]:
        """Make exactly the parameters in `names` require gradients; returns them."""
        names = set(names)
        known = [name for name, _ in self.named_weights()]
        if names - set(known):
            raise ValueError(
                f"unknown parameter(s) {sorted(names - set(known))}; this network has "
                f"{known}"
            )
        for name, p in self.named_weights():
            p.requires_grad_(name in names)
        return [p for name, p in self.named_weights() if name in names]

    def rates(self, v: torch.Tensor) -> torch.Tensor:
        """Rates for voltages `v` (..., n, steps)."""
        x = (v - E_L) / S
        return torch.relu(self.g)[:, None] * F.softplus(x + self.b[:, None])

    def edge_params(self) -> List[Dict[str, torch.Tensor]]:
        return self.sim.edge_params(self.W, self.g, self.b)


def new_network(
    model: BLKRate,
    init_io: Callable,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    compile: bool,
) -> Tuple[RateNetwork, Dict]:
    """The paper's initial network for `seed`, with the input and readout weights that
    `init_io(n, rng, alpha)` draws from the same random generator after the network."""
    rng = np.random.default_rng(seed)
    g, b, mask, w, n_exc = paper_init(
        model.neurons, rng, model.exc_frac, model.connection_prob
    )
    conn, w_edges = edges_from_mask(mask, w, n_exc)
    io = init_io(model.neurons, rng, model.dt / TAU)
    t0 = time.time()
    sim = Simulator(conn, device, dtype, model.dt, compile)
    print(
        f"network: {model.neurons} neurons ({n_exc} excitatory), "
        f"{int(conn['real'].sum()):,} synapses (padded to {conn['k']} per neuron), "
        f"built in {time.time() - t0:.1f} s",
        flush=True,
    )
    return RateNetwork(sim, w_edges, g, b, io), conn


def save_network(path: str, net: RateNetwork, conn: Dict, model: Dict, task: Dict):
    """The connectivity and parameters of a trained network, with its [model] and
    [task] settings."""
    torch.save(
        {
            "conn": {k: torch.as_tensor(v) for k, v in conn.items()},
            "state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "model": model,
            "task": task,
        },
        path,
    )


def _read_network(run_dir: str) -> Tuple[Dict, Dict, Dict]:
    """The saved network of a run directory, its connectivity and its settings: from
    `network.pt`, or from the `teacher.pt` of the scripts that ran
    examples/activitome-connectome/task-complexity before this library (they used the
    paper's settings; of the task, only what that file records)."""
    path = os.path.join(run_dir, "network.pt")
    old = not os.path.exists(path)
    if old:
        path = os.path.join(run_dir, "teacher.pt")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    conn = {k: (v.item() if v.dim() == 0 else v.numpy()) for k, v in ck["conn"].items()}
    if old:
        model = BLKRate(int(conn["n"]), 0.7, 0.5, 0.5).settings()
        task = (
            {"name": "cycling"}
            if ck["task"] == "cycling"
            else {"name": "text", "width": ck["args"]["width"], "steps_per_token": 10}
        )
        return ck, conn, {"model": model, "task": task}
    return ck, conn, {"model": ck["model"], "task": ck["task"]}


def network_settings(run_dir: str) -> Dict:
    """The [model] and [task] settings a run directory's network was trained with
    ({"model": ..., "task": ...})."""
    return _read_network(run_dir)[2]


def load_network(
    run_dir: str, device: torch.device, dtype: torch.dtype, compile: bool
) -> Tuple[RateNetwork, Dict, Dict]:
    """The trained network of a run directory (no parameter requires gradients), its
    connectivity and its settings (`network_settings`)."""
    ck, conn, settings = _read_network(run_dir)
    sim = Simulator(conn, device, dtype, settings["model"]["dt"], compile)
    st = ck["state"]
    io = {k[len("io.") :]: v for k, v in st.items() if k.startswith("io.")}
    return RateNetwork(sim, st["W"], st["g"], st["b"], io), conn, settings
