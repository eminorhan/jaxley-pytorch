# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""The rate network of Beiran & Litwin-Kumar (2025, Nat. Neurosci. 28:2561), selected by
`[model] name = "blk_rate"`.

Every neuron i has a state x_i and the output rate

    a_i = relu(g_i) * softplus(x_i + b_i)

with a gain g_i and a bias b_i. `Simulator` steps the network with the update of the
paper's code (forward Euler with time step dt, alpha = dt / tau, tau = 10 ms) and
computes the synaptic input of every step as one matrix product:

    x_i <- x_i + alpha * (-x_i + sum_j J_ij a_j + I_i)

with input currents I_i. It takes and returns membrane potentials v_i = E_L + S * x_i
(S = 10 mV per model unit), which `RateNetwork.rates` reads.
`tests/training/test_rate_network.py` checks it against the update above. The test also
simulates the same network in jaxley_pytorch (one passive compartment per neuron,
`SoftplusRateSynapse` synapses) and checks that against the update too.

Connectivity and initialisation are those of the paper's Fig. 1 teacher
(`nn_fig1_Teacher.py` in github.com/emebeiran/connconstr): a fraction `exc_frac` of the
neurons excitatory, every ordered pair connected with probability `connection_prob`
(autapses included), weights 2.4 / sqrt(N) * normal + 1 / N, inhibitory ones scaled by 4
(with `fan_in` instead: probability fan_in / N, and 2 * fan_in in place of N), Dale's
law as `relu(W * sign) * sign` (a weight that crosses zero stays at zero); gains ~
normal(1, 0.4) (positive), biases ~ uniform(-0.1, 0.1); the connectivity is drawn again
until the paper's stability test passes. The synapses are listed per postsynaptic neuron
and padded with zero-weight synapses to the largest in-degree (the layout of the saved
networks, in which jaxley_pytorch sums synaptic currents without atomic adds,
`Network._step_regular_synapses`).
"""

import os
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TAU = 10.0  # ms
E_L = -70.0  # mV (x = 0)
S = 10.0  # mV per unit of x


class BLKRate:
    """`[model] name = "blk_rate"`: `neurons` rate neurons, a fraction `exc_frac` of
    them excitatory, simulated with a time step of `dt` ms. Every ordered pair is
    connected with probability `connection_prob`, or, with `fan_in` instead (the
    expected number of inputs per neuron), with probability fan_in / neurons; set one
    of the two. With `fan_in`, the paper's weights are scaled for 2 * fan_in neurons
    instead of `neurons` (`paper_init`), so that a neuron's summed input has the
    statistics it has in the paper's networks (connection_prob = 0.5); fan_in =
    neurons / 2 draws the paper's network."""

    PARAMETERS = ("W", "g", "b")

    def __init__(
        self,
        neurons: int,
        exc_frac: float,
        dt: float,
        connection_prob: Optional[float] = None,
        fan_in: Optional[int] = None,
    ):
        if not (isinstance(neurons, int) and neurons >= 2):
            raise ValueError(
                f"[model] neurons must be an integer >= 2, got {neurons!r}"
            )
        if not 0.0 < exc_frac < 1.0:
            raise ValueError(f"[model] exc_frac must be in (0, 1), got {exc_frac!r}")
        if (connection_prob is None) == (fan_in is None):
            raise ValueError("[model] set exactly one of connection_prob and fan_in")
        if connection_prob is not None and not 0.0 < connection_prob <= 1.0:
            raise ValueError(
                f"[model] connection_prob must be in (0, 1], got {connection_prob!r}"
            )
        if fan_in is not None and not (
            isinstance(fan_in, int) and 1 <= fan_in <= neurons
        ):
            raise ValueError(
                f"[model] fan_in must be an integer in [1, neurons], got {fan_in!r}"
            )
        if not dt > 0.0:
            raise ValueError(f"[model] dt must be positive, got {dt!r}")
        self.neurons, self.exc_frac, self.dt = neurons, exc_frac, dt
        self.connection_prob, self.fan_in = connection_prob, fan_in

    @property
    def p(self) -> float:
        """The probability that an ordered pair is connected."""
        if self.fan_in is None:
            return self.connection_prob
        return self.fan_in / self.neurons

    @property
    def weight_n(self) -> int:
        """The N of the paper's weight scale (`paper_init`)."""
        return self.neurons if self.fan_in is None else 2 * self.fan_in

    def settings(self) -> Dict:
        if self.fan_in is None:
            connectivity = {"connection_prob": self.connection_prob}
        else:
            connectivity = {"fan_in": self.fan_in}
        return {
            "name": "blk_rate",
            "neurons": self.neurons,
            "exc_frac": self.exc_frac,
            **connectivity,
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


def paper_init(
    n: int,
    rng: np.random.Generator,
    exc_frac: float,
    p: float,
    weight_n: Optional[int] = None,
):
    """Gains, biases, connection mask (post, pre), weights and number of excitatory
    neurons of the paper's Fig. 1 teacher before training, every ordered pair connected
    with probability `p`; the connectivity is drawn again until `max_real_eigenvalue`
    <= 1. The weights are 2.4 / sqrt(weight_n) * normal + 1 / weight_n, with
    `weight_n` = n (the paper's) by default."""
    n_exc = int(exc_frac * n)
    weight_n = n if weight_n is None else weight_n
    g, b = random_gains_biases(n, rng)
    for _ in range(100):
        mask = rng.random((n, n)) < p
        w = 2.4 / np.sqrt(weight_n) * rng.standard_normal((n, n)) + 1.0 / weight_n
        w = w * mask
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


class _Synapses:
    """A connectivity (`edges_from_mask`) on `device`: every synapse's presynaptic and
    postsynaptic neuron, its sign (Dale's law) and whether it is real."""

    def __init__(self, conn: Dict, device: torch.device, dtype: torch.dtype):
        self.n, self.n_exc, self.k = int(conn["n"]), int(conn["n_exc"]), int(conn["k"])
        self.device, self.dtype = device, dtype
        pre = np.asarray(conn["pre"])

        def as_t(x, dt=dtype):
            return torch.as_tensor(np.asarray(x), device=device, dtype=dt)

        self.pre = as_t(pre, torch.long)
        self.post = as_t(conn["post"], torch.long)
        self.sign = as_t(np.where(pre < self.n_exc, 1.0, -1.0))
        self.real = as_t(conn["real"])

    def weights(self, W: torch.Tensor) -> torch.Tensor:
        """The weights `W` (per synapse) after Dale's law and the mask."""
        return torch.relu(W * self.sign) * self.sign * self.real

    def dense_weights(self, W: torch.Tensor) -> torch.Tensor:
        """J (post, pre) after Dale's law and the mask (the padding adds zeros)."""
        J = torch.zeros(self.n * self.n, device=W.device, dtype=W.dtype)
        J = J.index_add(0, self.post * self.n + self.pre, self.weights(W))
        return J.view(self.n, self.n)


def _update(v, current, JT, gain, b, alpha: float):
    """The steps of `Simulator.segment`: the voltages (batch, n, steps) after every step
    of `current` (batch, n, steps) from the voltages `v` (batch, n), and the last ones."""
    x = (v - E_L) / S
    xs = []
    for t in range(current.shape[-1]):
        a = gain * F.softplus(x + b)
        x = x + alpha * (-x + a @ JT + current[..., t])
        xs.append(x)
    return E_L + S * torch.stack(xs, dim=-1), E_L + S * x


class Simulator(_Synapses):
    """The network of a connectivity, shared by a teacher and its students (their
    parameters are passed to every simulation): the paper's update, the synaptic input
    of every step one matrix product (`_update`, `torch.compile`d if `compile`)."""

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
        self._update = torch.compile(_update, dynamic=False) if compile else _update

    def params(self, W: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> Dict:
        """What `segment` takes for weights `W` (per synapse), gains `g` and biases `b`
        (per neuron)."""
        return {"JT": self.dense_weights(W).T, "gain": torch.relu(g), "b": b}

    def rest_states(self, batch: int) -> Dict[str, torch.Tensor]:
        """Fresh states of `batch` networks with every neuron at rest (x = 0)."""
        v = torch.full((batch, self.n), E_L, device=self.device, dtype=self.dtype)
        return {"v": v}

    def segment(
        self, states: Dict[str, torch.Tensor], current: torch.Tensor, params: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """Simulate input currents `current` (batch, n, steps; uA/cm^2) from `states`.
        Returns the voltages after every step (batch, n, steps) and the final states."""
        v, last = self._update(
            states["v"],
            current,
            params["JT"],
            params["gain"],
            params["b"],
            self.dt / TAU,
        )
        return v, {"v": last}


class RateNetwork(nn.Module):
    """One network: weights `W` (per synapse), gains `g` and biases `b` (per neuron),
    and a task's input and readout weights `io` (see `tasks.py`), simulated by `sim`,
    which a teacher shares with its students. No parameter requires gradients until
    `set_trainable`."""

    neuron_param_names = ("g", "b")  # the single-neuron parameters

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

    def features(self, v: torch.Tensor) -> torch.Tensor:
        """The activity a task reads: the rates."""
        return self.rates(v)

    def neuron_params(self) -> Dict[str, torch.Tensor]:
        """The single-neuron parameters as they are compared with a teacher's: the
        gains (relu(g)) and the biases."""
        return {"gain": torch.relu(self.g), "bias": self.b}

    def sim_params(self) -> Dict:
        """What `self.sim.segment` takes for this network's weights, gains and biases."""
        return self.sim.params(self.W, self.g, self.b)


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
        model.neurons, rng, model.exc_frac, model.p, model.weight_n
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


def read_network(run_dir: str) -> Tuple[Dict, Dict, Dict]:
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
        model = BLKRate(int(conn["n"]), 0.7, dt=0.5, connection_prob=0.5).settings()
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
    return read_network(run_dir)[2]


def load_network(
    run_dir: str, device: torch.device, dtype: torch.dtype, compile: bool
) -> Tuple[RateNetwork, Dict, Dict]:
    """The trained network of a run directory (no parameter requires gradients), its
    connectivity and its settings (`network_settings`)."""
    ck, conn, settings = read_network(run_dir)
    sim = Simulator(conn, device, dtype, settings["model"]["dt"], compile)
    st = ck["state"]
    io = {k[len("io.") :]: v for k, v in st.items() if k.startswith("io.")}
    return RateNetwork(sim, st["W"], st["g"], st["b"], io), conn, settings
