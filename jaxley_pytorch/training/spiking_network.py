# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Networks of biophysical point neurons, selected by `[model] name = "pospischil"` or
`"hh"`: the neurons and synapses of examples/language_modeling, with trainable
single-neuron parameters.

Every neuron is one compartment (a cylinder of 10 um radius and 20 um length) with the
channels of its type:

  pospischil  excitatory neurons regular spiking (Na, K, M-type K, leak), inhibitory
              ones fast spiking (Na, K, leak), as in Pospischil et al. (2008);
  hh          Hodgkin-Huxley (Na, K, leak), every neuron.

A fraction `exc_frac` of the neurons is excitatory, and every neuron receives `fan_in`
synapses, a fraction `exc_frac` of them from random excitatory neurons and the rest
from random inhibitory ones (no autapses). The synapses are conductance-based
first-order kinetic synapses (`jaxley_pytorch.synapses.KineticSynapse`): reversal
potential 0 mV (excitatory) or -80 mV (inhibitory), rise rate 1/ms, decay time constant
5 ms (excitatory) or 10 ms (inhibitory), transmitter released above -20 mV (slope
`release_width`, 2 mV by default), i.e. by spikes. A synapse's conductance is softplus(W) times the scale of its type
(`w_exc`, `w_inh`: mS/cm^2 per synapse at a fan-in of 100, times 100 / fan_in), so it
is never negative and keeps the sign of its type (Dale's law). The synapses are ordered
by postsynaptic neuron, `fan_in` each, the layout for which jaxley_pytorch uses its
fused synapse kernels on CUDA.

A neuron's single-neuron parameters are the maximal conductances of its channels,
trained as their logarithms (they span orders of magnitude), and a bias current. The
teacher's conductances are drawn around the defaults of its type (log-normal with
standard deviation `spread`); its biases start at 0. The task's input current, times
`input_scale`, and the bias are densities (uA/cm^2). The activity a task reads is every
neuron's voltage, centred at -60 mV and scaled by 20 mV (`SpikingNetwork.features`);
spikes are upward crossings of 0 mV (`SpikingNetwork.spike_counts`).

With `na_activation_gradient` below 1 (1, the exact gradient, by default), the gradient
is a surrogate: in the backward pass only, the sodium channels' activation gate m reads
the voltage with that derivative instead of 1. The voltage's value is unchanged, and so
are the simulation, its spikes and the loss. This weakens the spike upstroke's feedback
loop (voltage -> m -> sodium current -> voltage) in the gradient; the exact gradient of
a long simulation changes direction under tiny changes of the parameters.
"""

import math
import os
import time
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH, K, Km, Leak, Na
from jaxley_pytorch.synapses import KineticSynapse
from jaxley_pytorch.training.rate_network import read_network

RADIUS, LENGTH = 10.0, 20.0  # um, point neurons
AREA_CM2 = 2 * math.pi * RADIUS * LENGTH * 1e-8
PER_DENSITY = AREA_CM2 * 1e3  # nA per uA/cm^2, and uS per mS/cm^2
V_REST, V_CENTER, V_SCALE, SPIKE_V = -65.0, -60.0, 20.0, 0.0  # mV
E_EXC, E_INH = 0.0, -80.0  # mV, synaptic reversal potentials
ALPHA, BETA_EXC, BETA_INH, V_TH, DELTA = 1.0, 0.2, 0.1, -20.0, 2.0  # 1/ms, mV

# Every model's cell types: the channels and the values set on them (excitatory,
# inhibitory), and its trainable conductances (name -> channel parameter, and whether
# only the excitatory neurons have it).
CELLS = {
    "pospischil": (
        (
            (Na, K, Km, Leak),
            {
                "Na_gNa": 5e-2,
                "K_gK": 5e-3,
                "Km_gKm": 7e-5,
                "Leak_gLeak": 1e-4,
                "Km_taumax": 1000.0,
                "vt": -60.0,
                "Leak_eLeak": -70.0,
            },
        ),
        (
            (Na, K, Leak),
            {
                "Na_gNa": 5e-2,
                "K_gK": 1e-2,
                "Leak_gLeak": 1.5e-4,
                "vt": -60.0,
                "Leak_eLeak": -70.0,
            },
        ),
    ),
    "hh": (
        ((HH,), {"HH_gNa": 0.12, "HH_gK": 0.036, "HH_gLeak": 3e-4}),
        ((HH,), {"HH_gNa": 0.12, "HH_gK": 0.036, "HH_gLeak": 3e-4}),
    ),
}
CONDUCTANCES = {
    "pospischil": {
        "gNa": ("Na_gNa", False),
        "gK": ("K_gK", False),
        "gKm": ("Km_gKm", True),
        "gLeak": ("Leak_gLeak", False),
    },
    "hh": {
        "gNa": ("HH_gNa", False),
        "gK": ("HH_gK", False),
        "gLeak": ("HH_gLeak", False),
    },
}


class SpikingModel:
    """`[model] name = "pospischil"` or `"hh"` (see the module's docstring): `neurons`
    neurons, a fraction `exc_frac` of them excitatory, `fan_in` synapses each, simulated
    with a time step of `dt` ms; `input_scale` (uA/cm^2 per unit of the task's input),
    `w_exc` and `w_inh` (mS/cm^2 per synapse at a fan-in of 100), `spread` (of the
    teacher's log conductances), `release_width` (mV, the slope of the synapses'
    transmitter release around -20 mV) and `na_activation_gradient` (from 0 to 1: the
    derivative with which the sodium activation gate reads the voltage in the backward
    pass; 1 is the exact gradient, below 1 a surrogate, see the module's docstring)."""

    name = ""
    PARAMETERS: Tuple[str, ...] = ()

    def __init__(
        self,
        neurons: int,
        exc_frac: float,
        fan_in: int,
        dt: float,
        input_scale: float,
        w_exc: float,
        w_inh: float,
        spread: float,
        release_width: float = DELTA,
        na_activation_gradient: float = 1.0,
    ):
        if not (isinstance(neurons, int) and neurons >= 2):
            raise ValueError(
                f"[model] neurons must be an integer >= 2, got {neurons!r}"
            )
        if not 0.0 < exc_frac < 1.0:
            raise ValueError(f"[model] exc_frac must be in (0, 1), got {exc_frac!r}")
        self.neurons, self.exc_frac, self.fan_in = neurons, exc_frac, fan_in
        if not (isinstance(fan_in, int) and fan_in >= 1):
            raise ValueError(f"[model] fan_in must be an integer >= 1, got {fan_in!r}")
        if (
            self.k_exc > self.n_exc - 1
            or fan_in - self.k_exc > neurons - self.n_exc - 1
        ):
            raise ValueError(
                f"[model] fan_in = {fan_in} needs more neurons of each type than "
                f"{self.n_exc} excitatory and {neurons - self.n_exc} inhibitory"
            )
        positive = (("dt", dt), ("w_exc", w_exc), ("w_inh", w_inh))
        for key, value in positive + (("release_width", release_width),):
            if not value > 0.0:
                raise ValueError(f"[model] {key} must be positive, got {value!r}")
        for key, value in (("input_scale", input_scale), ("spread", spread)):
            if not value >= 0.0:
                raise ValueError(f"[model] {key} must be >= 0, got {value!r}")
        if not 0.0 <= na_activation_gradient <= 1.0:
            raise ValueError(
                "[model] na_activation_gradient must be from 0 to 1, got "
                f"{na_activation_gradient!r}"
            )
        self.dt, self.input_scale = dt, input_scale
        self.w_exc, self.w_inh, self.spread = w_exc, w_inh, spread
        self.release_width = release_width
        self.na_activation_gradient = float(na_activation_gradient)

    @property
    def n_exc(self) -> int:
        return int(round(self.exc_frac * self.neurons))

    @property
    def k_exc(self) -> int:
        """Excitatory synapses per neuron."""
        return int(round(self.exc_frac * self.fan_in))

    def settings(self) -> Dict:
        return {
            "name": self.name,
            "neurons": self.neurons,
            "exc_frac": self.exc_frac,
            "fan_in": self.fan_in,
            "dt": self.dt,
            "input_scale": self.input_scale,
            "w_exc": self.w_exc,
            "w_inh": self.w_inh,
            "spread": self.spread,
            "release_width": self.release_width,
            "na_activation_gradient": self.na_activation_gradient,
        }


class Pospischil(SpikingModel):
    name = "pospischil"
    PARAMETERS = ("W", "gNa", "gK", "gKm", "gLeak", "bias")


class HodgkinHuxley(SpikingModel):
    name = "hh"
    PARAMETERS = ("W", "gNa", "gK", "gLeak", "bias")


SPIKING_MODELS = {m.name: m for m in (Pospischil, HodgkinHuxley)}


def sample_connectivity(
    n: int, n_exc: int, fan_in: int, k_exc: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """For every neuron, `k_exc` distinct random excitatory and `fan_in - k_exc`
    inhibitory presynaptic neurons other than itself. Returns (pre, post), ordered by
    postsynaptic neuron (its excitatory inputs first)."""
    pre, post = [], []
    for pool, k in ((np.arange(n_exc), k_exc), (np.arange(n_exc, n), fan_in - k_exc)):
        if k == 0:
            continue
        for j in range(n):
            cand = rng.choice(pool, size=k + 1, replace=False)
            pre.append(cand[cand != j][:k])
            post.append(np.full(k, j))
    pre, post = np.concatenate(pre), np.concatenate(post)
    order = np.argsort(post, kind="stable")
    return pre[order], post[order]


def _soft_voltage(v: torch.Tensor, gradient: float) -> torch.Tensor:
    """`v` (the same value) with the derivative `gradient`."""
    return v.detach() + gradient * (v - v.detach())


class _SurrogateNa(Na):
    """`Na` whose activation gate reads the voltage with the derivative `gradient`."""

    def __init__(self, gradient: float):
        super().__init__("Na")
        self.gradient = gradient

    def m_gate(self, v, vt):
        return Na.m_gate(_soft_voltage(v, self.gradient), vt)


class _SurrogateHH(HH):
    """`HH` whose sodium activation gate reads the voltage with the derivative
    `gradient`."""

    def __init__(self, gradient: float):
        super().__init__("HH")
        self.gradient = gradient

    def m_gate(self, v):
        return HH.m_gate(_soft_voltage(v, self.gradient))


SURROGATES = {Na: _SurrogateNa, HH: _SurrogateHH}


def _point_cell(channels, values: Dict, na_activation_gradient: float = 1.0) -> jx.Cell:
    cell = jx.Cell()
    cell.set("radius", RADIUS)
    cell.set("length", LENGTH)
    for channel in channels:
        if na_activation_gradient != 1.0 and channel in SURROGATES:
            cell.insert(SURROGATES[channel](na_activation_gradient))
        else:
            cell.insert(channel())
    for key, value in values.items():
        cell.set(key, value)
    return cell


def default_conductances(model: SpikingModel) -> Dict[str, np.ndarray]:
    """Every trainable conductance's value of its type, for the neurons that have it."""
    (_, exc), (_, inh) = CELLS[model.name]
    n, n_exc = model.neurons, model.n_exc
    out = {}
    for name, (key, exc_only) in CONDUCTANCES[model.name].items():
        values = np.full(n, exc[key], dtype=float)
        values[n_exc:] = inh.get(key, np.nan)
        out[name] = values[:n_exc] if exc_only else values
    return out


class SpikingSimulator:
    """The jaxley_pytorch network of a connectivity, shared by a teacher and its
    students (their parameters are passed to every simulation)."""

    def __init__(
        self,
        model: SpikingModel,
        conn: Dict,
        device: torch.device,
        dtype: torch.dtype,
        compile: bool = True,
    ):
        self.n, self.n_exc, self.k = int(conn["n"]), int(conn["n_exc"]), int(conn["k"])
        self.device, self.dtype, self.dt, self.compile = (
            device,
            dtype,
            model.dt,
            compile,
        )
        self.input_scale = model.input_scale
        (exc_channels, exc), (inh_channels, inh) = CELLS[model.name]
        exc_cell, inh_cell = (
            _point_cell(exc_channels, exc, model.na_activation_gradient),
            _point_cell(inh_channels, inh, model.na_activation_gradient),
        )
        net = jx.Network([exc_cell] * self.n_exc + [inh_cell] * (self.n - self.n_exc))
        nodes = net.nodes
        first = nodes[
            (nodes["local_branch_index"] == 0) & (nodes["local_comp_index"] == 0)
        ]
        first = first.sort_values("global_cell_index").index.to_numpy()
        assert len(first) == self.n
        pre, post = np.asarray(conn["pre"]), np.asarray(conn["post"])
        synapse = KineticSynapse()
        net._append_multiple_synapses(
            net.select(nodes=first[pre]).nodes,
            net.select(nodes=first[post]).nodes,
            synapse,
        )
        name = synapse._name
        is_exc = pre < self.n_exc
        net.edges[f"{name}_e_syn"] = np.where(is_exc, E_EXC, E_INH)
        net.edges[f"{name}_alpha"] = ALPHA
        net.edges[f"{name}_beta"] = np.where(is_exc, BETA_EXC, BETA_INH)
        net.edges[f"{name}_v_th"] = V_TH
        net.edges[f"{name}_delta"] = model.release_width
        net.set("v", V_REST)
        net.init_states()
        getattr(net, name).edge("all").make_trainable(f"{name}_gS", verbose=False)
        # One parameter per neuron that has the channel, in neuron order.
        self.conductances = [
            (cond, key) for cond, (key, _) in CONDUCTANCES[model.name].items()
        ]
        for _, key in self.conductances:
            net.cell("all").branch("all").make_trainable(key, verbose=False)
        net.cell("all").branch(0).comp(0).record("v", verbose=False)
        self.model, self.net, self.name = model, net, name
        self.input_view = net.select(nodes=first)
        scale = np.where(is_exc, model.w_exc, model.w_inh) * (100.0 / self.k)
        self.w_scale = torch.as_tensor(scale * PER_DENSITY, device=device, dtype=dtype)
        self.real = torch.ones(len(pre))  # every synapse is real (no padding)
        self._template = {}

    def params(
        self, W: torch.Tensor, log_g: Dict[str, torch.Tensor], bias: torch.Tensor
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        """What `segment` takes for weights `W` (per synapse), log conductances `log_g`
        and biases `bias` (per neuron): the parameters of the jaxley network (in
        `make_trainable` order) and the bias current."""
        gs = F.softplus(W) * self.w_scale
        jaxley = [{f"{self.name}_gS": gs}]
        jaxley += [{key: torch.exp(log_g[cond])} for cond, key in self.conductances]
        return jaxley, bias

    def rest_states(self, batch: int) -> Dict[str, torch.Tensor]:
        """Fresh states of `batch` networks at rest (-65 mV, the gates at their steady
        state, the synapses closed)."""
        if batch not in self._template:
            zeros = torch.zeros(batch, self.n, 1, device=self.device, dtype=self.dtype)
            with torch.no_grad():
                _, states = jx.integrate(
                    self.net,
                    data_stimuli=self.input_view.data_stimulate(zeros),
                    delta_t=self.dt,
                    return_states=True,
                )
            self._template[batch] = {k: v.detach() for k, v in states.items()}
        return {
            k: torch.full_like(v, V_REST) if k == "v" else v.clone()
            for k, v in self._template[batch].items()
        }

    def segment(
        self, states: Dict[str, torch.Tensor], current: torch.Tensor, params
    ) -> Tuple[torch.Tensor, Dict]:
        """Simulate the task's input currents `current` (batch, n, steps) from `states`.
        Returns the voltages after every step (batch, n, steps; mV) and the final
        states."""
        jaxley, bias = params
        density = current * self.input_scale + bias[:, None]  # uA/cm^2
        v, states = jx.integrate(
            self.net,
            params=jaxley,
            data_stimuli=self.input_view.data_stimulate(density * PER_DENSITY),
            all_states=states,
            return_states=True,
            delta_t=self.dt,
            compile=self.compile,
        )
        return v[..., 1:], states


class SpikingNetwork(nn.Module):
    """One network: weights `W` (per synapse), log conductances `log_g` and biases
    `bias` (per neuron), and a task's input and readout weights `io`, simulated by `sim`,
    which a teacher shares with its students. No parameter requires gradients until
    `set_trainable`."""

    def __init__(self, sim: SpikingSimulator, W, log_g: Dict, bias, io: Dict):
        super().__init__()
        self.sim = sim
        self.W = self._param(W)
        # (An OrderedDict keeps the model's order; a ParameterDict sorts a dict's keys.)
        self.log_g = nn.ParameterDict(
            OrderedDict((k, self._param(v)) for k, v in log_g.items())
        )
        self.bias = self._param(bias)
        self.io = nn.ParameterDict({k: self._param(v) for k, v in io.items()})

    def _param(self, value) -> nn.Parameter:
        t = torch.as_tensor(value).to(device=self.sim.device, dtype=self.sim.dtype)
        return nn.Parameter(t.clone(), requires_grad=False)

    @property
    def neuron_param_names(self) -> Tuple[str, ...]:
        """The single-neuron parameters: the log conductances and the bias."""
        return (*self.log_g, "bias")

    def named_weights(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """(name, parameter) for "W", the log conductances, "bias" and the task's input
        and readout weights, in this order."""
        yield ("W", self.W)
        yield from self.log_g.items()
        yield ("bias", self.bias)
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

    def features(self, v: torch.Tensor) -> torch.Tensor:
        """The activity a task reads for voltages `v` (..., n, steps)."""
        return (v - V_CENTER) / V_SCALE

    def spike_counts(self, v_before: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Spikes of every neuron in the voltages `v` (..., n, steps), `v_before`
        (..., n) the voltage before them: upward crossings of 0 mV."""
        above = torch.cat([v_before[..., None], v], dim=-1) >= SPIKE_V
        return (above[..., 1:] & ~above[..., :-1]).sum(-1).to(v.dtype)

    def neuron_params(self) -> Dict[str, torch.Tensor]:
        """The single-neuron parameters as they are compared with a teacher's."""
        return {**self.log_g, "bias": self.bias}

    def sim_params(self):
        """What `self.sim.segment` takes for this network's parameters."""
        return self.sim.params(self.W, dict(self.log_g), self.bias)


def new_network(
    model: SpikingModel,
    init_io: Callable,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    compile: bool,
) -> Tuple[SpikingNetwork, Dict]:
    """The initial network for `seed`: connectivity, weights (softplus(W) log-normal
    around 1), log conductances around the defaults of every type, biases 0, and the
    input and readout weights that `init_io(n, rng, alpha)` draws from the same random
    generator after them."""
    rng = np.random.default_rng(seed)
    n, n_exc = model.neurons, model.n_exc
    pre, post = sample_connectivity(n, n_exc, model.fan_in, model.k_exc, rng)
    W = np.log(np.expm1(np.exp(rng.normal(0.0, 0.5, len(pre)))))
    log_g = {
        name: np.log(values) + model.spread * rng.standard_normal(len(values))
        for name, values in default_conductances(model).items()
    }
    io = init_io(n, rng, 1.0)
    conn = {
        "n": n,
        "n_exc": n_exc,
        "k": model.fan_in,
        "pre": pre,
        "post": post,
        "real": np.ones(len(pre)),
    }
    t0 = time.time()
    sim = SpikingSimulator(model, conn, device, dtype, compile)
    print(
        f"network: {n} {model.name} neurons ({n_exc} excitatory), {len(pre):,} synapses "
        f"({model.fan_in} per neuron), built in {time.time() - t0:.1f} s",
        flush=True,
    )
    return SpikingNetwork(sim, W, log_g, np.zeros(n), io), conn


def load_network(
    run_dir: str, device: torch.device, dtype: torch.dtype, compile: bool
) -> Tuple[SpikingNetwork, Dict, Dict]:
    """The trained network of a run directory (no parameter requires gradients), its
    connectivity and its settings."""
    ck, conn, settings = read_network(run_dir)
    kwargs = {k: v for k, v in settings["model"].items() if k != "name"}
    model = SPIKING_MODELS[settings["model"]["name"]](**kwargs)
    sim = SpikingSimulator(model, conn, device, dtype, compile)
    st = ck["state"]
    io = {k[len("io.") :]: v for k, v in st.items() if k.startswith("io.")}
    log_g = {k[len("log_g.") :]: v for k, v in st.items() if k.startswith("log_g.")}
    return SpikingNetwork(sim, st["W"], log_g, st["bias"], io), conn, settings


def make_student(
    teacher: SpikingNetwork,
    conn: Dict,
    start: str,
    seed: int,
    student_dir: Optional[str] = None,
) -> SpikingNetwork:
    """A network with the teacher's weights, inputs and readout and the single-neuron
    parameters of `start`: "shuffled" (the teacher's, every parameter permuted within
    the excitatory and within the inhibitory neurons, drawn with `seed`), "random" (new
    draws from the teacher's initial distribution with `seed`: log conductances around
    the defaults of every type, biases 0), "teacher" (its own) or "student" (those of
    the trained student in `student_dir`)."""
    values = {
        name: p.detach().cpu().double().numpy().copy()
        for name, p in teacher.neuron_params().items()
    }
    rng = np.random.default_rng(seed)
    n, n_exc = int(conn["n"]), int(conn["n_exc"])
    if start == "shuffled":
        for v in values.values():
            parts = (
                (slice(0, n_exc), slice(n_exc, None)) if len(v) == n else (slice(None),)
            )
            for part in parts:
                v[part] = rng.permutation(v[part])
    elif start == "random":
        model = teacher.sim.model
        values = {
            name: np.log(v) + model.spread * rng.standard_normal(len(v))
            for name, v in default_conductances(model).items()
        }
        values["bias"] = np.zeros(n)
    elif start == "student":
        path = os.path.join(student_dir, "student.pt")
        st = torch.load(path, map_location="cpu", weights_only=True)
        values = {name: st[name].double().numpy() for name in values}
    elif start != "teacher":
        raise ValueError(
            f'[init] start = "{start}" is not available for spiking networks '
            '("shuffled", "random", "teacher" or "student")'
        )
    io = {k: v.detach() for k, v in teacher.io.items()}
    log_g = {name: values[name] for name in teacher.log_g}
    return SpikingNetwork(teacher.sim, teacher.W.detach(), log_g, values["bias"], io)
