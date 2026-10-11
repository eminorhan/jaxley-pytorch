"""Biophysical recurrent networks for character-level language modelling.

Every network has `n` neurons, a fraction `exc_frac` of them excitatory, with sparse
recurrent synapses that respect Dale's law: every neuron receives `k_in` synapses
(`exc_frac * k_in` from random excitatory and the rest from random inhibitory
neurons, no autapses), and all synapses of a neuron have the sign of its type. The
trainable synaptic weights are kept non-negative by a softplus (the sign comes from
the synapse type, see below).

Neuron models (`MODELS`):

* `rate`: one passive compartment (leak, 10 ms membrane time constant) per neuron, and
  current-based synapses whose presynaptic "rate" is a sigmoid of the presynaptic
  voltage (`SigmoidRateSynapse`, below). Jaxley's `TanhRateSynapse` uses tanh, whose
  output is negative below its offset, so an excitatory weight would then inhibit;
  the sigmoid keeps rates non-negative so that the weight's sign alone fixes the
  synapse's sign (Dale's law).
* `hh`: one Hodgkin-Huxley compartment per neuron (`jx.channels.HH`, default
  parameters).
* `pospischil`: one compartment per neuron with the cortical channels of Pospischil et
  al. (2008): regular-spiking excitatory (Na, K, M-type K, leak) and fast-spiking
  inhibitory (Na, K, leak) neurons.
* `l5pc`: the three-compartment unit (apical, soma, basal) with L5PC channels of the
  Jaxley paper's RNNs (Fig. 4, `examples/paper/rnn_utils.py`); synapses run from the
  soma to the apical compartment, the input current enters the basal compartment, the
  readout uses the soma.

The spiking models use conductance-based first-order kinetic synapses (`KineticSynapse`,
below) with reversal potentials of 0 mV (excitatory) and -80 mV (inhibitory), a rise
rate of 1/ms and decay time constants of 5 ms and 10 ms; transmitter is released when
the presynaptic voltage exceeds -20 mV (slope 2 mV), i.e. by spikes. (Unlike Jaxley's
`IonotropicSynapse`, whose rise rate grows without bound with the presynaptic voltage,
the rise rate is bounded.)

Point neurons are cylinders of 10 um radius and 20 um length (1,257 um^2). Input
currents and synaptic weights are specified as densities (uA/cm^2, mS/cm^2) and
converted with the membrane area of the compartment they act on.

`gru` is a conventional GRU (dense) with the same number of units, as a reference.
"""
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH, K, Km, Leak, Na
from jaxley_pytorch.synapses.synapse import Synapse

MODELS = ("gru", "rate", "hh", "pospischil", "l5pc")
RADIUS, LENGTH = 10.0, 20.0  # um, point neurons.

# Per model: solver step (ms; 0.025 ms for the spiking models as in the Jaxley paper: at
# 0.05 ms spike times drift and the gradients through the recurrent dynamics grow much
# faster, see README), input current scale (uA/cm^2; chosen so that the untrained
# networks driven by text fire at ~25-45 Hz, or have mean rates ~0.3 for `rate`, see
# `train.py --mode activity`), synaptic weight scales
# for excitatory / inhibitory synapses (per synapse, multiplied by 100 / k_in;
# uA/cm^2 per unit rate for `rate`, mS/cm^2 otherwise), the voltage around which the
# readout features are centred and their scale (mV).
DEFAULTS = {
    "rate": dict(dt=0.5, input_scale=4.0, w_exc=0.05, w_inh=0.2, v_center=-60.0, v_scale=10.0),
    "hh": dict(dt=0.025, input_scale=6.0, w_exc=0.02, w_inh=0.1, v_center=-60.0, v_scale=20.0),
    "pospischil": dict(dt=0.025, input_scale=2.0, w_exc=0.01, w_inh=0.05, v_center=-60.0, v_scale=20.0),
    "l5pc": dict(dt=0.025, input_scale=20.0, w_exc=0.05, w_inh=0.25, v_center=-60.0, v_scale=20.0),
}


class SigmoidRateSynapse(Synapse):
    """Current-based rate synapse: `current = -gS * sigmoid((v_pre - x_offset) * slope)`.

    The presynaptic rate is non-negative, so the sign of `gS` alone sets the sign of
    the synapse: `gS > 0` depolarizes (excitatory), `gS < 0` hyperpolarizes. Like
    `TanhRateSynapse`, `gS` is a current (nA), it has no state.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,
            f"{prefix}_x_offset": -60.0,
            f"{prefix}_slope": 0.25,
        }
        self.synapse_states = {}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        return {}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        rate = torch.sigmoid((pre_voltage - params[f"{prefix}_x_offset"]) * params[f"{prefix}_slope"])
        return -params[f"{prefix}_gS"] * rate


class KineticSynapse(Synapse):
    """First-order kinetic synapse (Destexhe, Mainen & Sejnowski, 1994).

    `ds/dt = alpha * T(v_pre) * (1 - s) - beta * s` with the transmitter
    `T(v) = sigmoid((v - v_th) / delta)`, solved with exponential Euler, and
    `current = gS * s * (v_post - e_syn)` (nA for gS in uS). The time constant,
    `1 / (alpha * T + beta)`, stays bounded however steep the release is.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_e_syn": 0.0,  # mV
            f"{prefix}_alpha": 1.0,  # 1/ms
            f"{prefix}_beta": 0.2,  # 1/ms
            f"{prefix}_v_th": -20.0,  # mV
            f"{prefix}_delta": 2.0,  # mV
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        prefix = self._name
        transmitter = torch.sigmoid((pre_voltage - params[f"{prefix}_v_th"]) / params[f"{prefix}_delta"])
        a = params[f"{prefix}_alpha"] * transmitter
        rate = a + params[f"{prefix}_beta"]
        s_inf = a / rate
        s = states[f"{prefix}_s"]
        return {f"{prefix}_s": s_inf + (s - s_inf) * torch.exp(-delta_t * rate)}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        return params[f"{prefix}_gS"] * states[f"{prefix}_s"] * (post_voltage - params[f"{prefix}_e_syn"])


def _point_cell(channels) -> jx.Cell:
    cell = jx.Cell()
    cell.set("radius", RADIUS)
    cell.set("length", LENGTH)
    for ch in channels:
        cell.insert(ch)
    return cell


def _cells(model: str):
    """(excitatory cell, inhibitory cell, branch of the synapse source, branch of the
    synapse target and input, branch of the readout)."""
    if model == "rate":
        cell = _point_cell([Leak()])
        cell.set("Leak_gLeak", 1e-4)  # 10 ms with 1 uF/cm^2.
        cell.set("Leak_eLeak", -70.0)
        return cell, cell, 0, 0, 0
    if model == "hh":
        cell = _point_cell([HH()])
        return cell, cell, 0, 0, 0
    if model == "pospischil":
        rs = _point_cell([Na(), K(), Km(), Leak()])
        rs.set("K_gK", 5e-3)
        rs.set("Km_gKm", 7e-5)
        rs.set("Km_taumax", 1000.0)
        rs.set("Leak_gLeak", 1e-4)
        fs = _point_cell([Na(), K(), Leak()])
        fs.set("K_gK", 1e-2)
        fs.set("Leak_gLeak", 1.5e-4)
        for c in (rs, fs):
            c.set("Na_gNa", 5e-2)
            c.set("vt", -60.0)
            c.set("Leak_eLeak", -70.0)
        return rs, fs, 0, 0, 0
    if model == "l5pc":
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper"))
        from rnn_utils import build_unit

        unit = build_unit()
        # Synapses soma -> apical (as in the paper); input into the basal compartment.
        return unit, unit, 1, 0, 1
    raise ValueError(model)


def _first_nodes(net: jx.Network, branch: int) -> np.ndarray:
    """Node index of compartment 0 of `branch` of every cell, in cell order."""
    nodes = net.nodes
    sel = nodes[(nodes["local_branch_index"] == branch) & (nodes["local_comp_index"] == 0)]
    sel = sel.sort_values("global_cell_index")
    assert len(sel) == nodes["global_cell_index"].nunique()
    return sel.index.to_numpy()


def _area_cm2(net: jx.Network, node_inds: np.ndarray) -> np.ndarray:
    nodes = net.nodes.loc[node_inds]
    return 2 * np.pi * nodes["radius"].to_numpy() * nodes["length"].to_numpy() * 1e-8


def sample_connectivity(n: int, exc_frac: float, k_in: int, rng: np.random.Generator):
    """Fixed in-degree E/I connectivity without autapses. Returns (pre, post) cells."""
    n_exc = int(round(exc_frac * n))
    k_exc = int(round(exc_frac * k_in))
    k_inh = k_in - k_exc
    exc, inh = np.arange(n_exc), np.arange(n_exc, n)
    pre, post = [], []
    for pool, k in [(exc, k_exc), (inh, k_inh)]:
        # For every postsynaptic cell, k distinct presynaptic cells of the pool (other
        # than itself): sample k + 1 and drop the cell itself if drawn.
        draws = np.argsort(rng.random((n, len(pool))), axis=1)[:, : k + 1] if len(pool) <= 4096 else None
        for j in range(n):
            cand = pool[draws[j]] if draws is not None else rng.choice(pool, size=k + 1, replace=False)
            cand = cand[cand != j][:k]
            pre.append(cand)
            post.append(np.full(len(cand), j))
    return np.concatenate(pre), np.concatenate(post), n_exc


class BiophysNet:
    """A jaxley-pytorch network of `n` neurons of type `model` with sparse E/I synapses."""

    def __init__(self, model: str, n: int, k_in: int = 100, exc_frac: float = 0.8, seed: int = 0, verbose=True):
        self.model, self.n, self.k_in = model, n, k_in
        rng = np.random.default_rng(seed)
        exc_cell, inh_cell, src_branch, tgt_branch, read_branch = _cells(model)
        pre, post, n_exc = sample_connectivity(n, exc_frac, k_in, rng)
        self.n_exc = n_exc
        net = jx.Network([exc_cell] * n_exc + [inh_cell] * (n - n_exc))
        src = _first_nodes(net, src_branch)
        tgt = _first_nodes(net, tgt_branch)
        read = _first_nodes(net, read_branch)
        synapse = SigmoidRateSynapse() if model == "rate" else KineticSynapse()
        net._append_multiple_synapses(net.select(nodes=src[pre]).nodes, net.select(nodes=tgt[post]).nodes, synapse)
        is_exc = pre < n_exc
        name = synapse._name
        edges = net.edges
        if model == "rate":
            edges[f"{name}_x_offset"] = -60.0
            edges[f"{name}_slope"] = 0.25
        else:
            edges[f"{name}_e_syn"] = np.where(is_exc, 0.0, -80.0)
            edges[f"{name}_alpha"] = 1.0
            edges[f"{name}_beta"] = np.where(is_exc, 0.2, 0.1)
            edges[f"{name}_v_th"] = -20.0
            edges[f"{name}_delta"] = 2.0
        net.set("v", -70.0 if model == "rate" else -65.0)
        net.init_states()
        getattr(net, name).edge("all").make_trainable(f"{name}_gS", verbose=False)
        net.cell("all").branch(read_branch).comp(0).record("v", verbose=False)
        self.net, self.synapse_name = net, name
        self.pre, self.post, self.is_exc = pre, post, is_exc
        self.input_view = net.select(nodes=tgt if model != "l5pc" else _first_nodes(net, 2))
        # Unit conversions: inputs in uA/cm^2 -> nA on the input compartment; weights
        # in mS/cm^2 (uA/cm^2 per unit rate) -> uS (nA) on the target compartment.
        self.input_nA = torch.as_tensor(_area_cm2(net, self.input_view._nodes_in_view) * 1e3)
        self.weight_unit = torch.as_tensor(_area_cm2(net, tgt[post]) * 1e3)
        self.sign = torch.as_tensor(np.where(is_exc, 1.0, -1.0) if model == "rate" else np.ones(len(pre)))
        if verbose:
            print(f"{model}: {n} neurons ({n_exc} excitatory), {len(pre):,} synapses, "
                  f"{len(net.nodes)} compartments", flush=True)


class LanguageModel(nn.Module):
    """Character-level language model around a biophysical network (or a GRU).

    Every token is presented for `steps_per_token` solver steps as a constant current,
    `embedding[token]`, into the input compartment of every neuron. The readout is a
    linear map of each neuron's readout-compartment voltage, averaged over the token's
    window (centred and scaled), to next-token logits. The network state is carried
    from token to token and from chunk to chunk; each token's simulation is
    checkpointed (recomputed in the backward pass). With `bptt_tokens`, the gradient
    through the network state is also truncated every `bptt_tokens` tokens.
    """

    def __init__(self, model: str, vocab: int, n: int, steps_per_token: int, k_in: int = 100,
                 exc_frac: float = 0.8, dt: Optional[float] = None, input_scale: Optional[float] = None,
                 w_exc: Optional[float] = None, w_inh: Optional[float] = None, seed: int = 0,
                 compile: bool = True, inner_checkpoint: Optional[List[int]] = None, bptt_tokens: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.model, self.vocab, self.n, self.spt = model, vocab, n, steps_per_token
        self.compile, self.inner_checkpoint, self.bptt_tokens = compile, inner_checkpoint, bptt_tokens
        if model == "gru":
            self.embed = nn.Embedding(vocab, n)
            self.rnn = nn.GRU(n, n, batch_first=True)
            self.readout = nn.Linear(n, vocab)
            return
        cfg = dict(DEFAULTS[model])
        for k, v in dict(dt=dt, input_scale=input_scale, w_exc=w_exc, w_inh=w_inh).items():
            if v is not None:
                cfg[k] = v
        self.cfg = cfg
        self.bnet = BiophysNet(model, n, k_in=k_in, exc_frac=exc_frac, seed=seed)
        n_syn = len(self.bnet.pre)
        # Weights: softplus(theta) * scale, scale per synapse type (and k_in).
        scale = np.where(self.bnet.is_exc, cfg["w_exc"], cfg["w_inh"]) * (100.0 / k_in)
        self.register_buffer("w_scale", torch.as_tensor(scale) * self.bnet.weight_unit * self.bnet.sign)
        init = np.exp(np.random.default_rng(seed + 1).normal(0.0, 0.5, n_syn))  # log-normal around 1
        self.theta = nn.Parameter(torch.as_tensor(np.log(np.expm1(init))))
        self.register_buffer("input_nA", self.bnet.input_nA.clone())
        self.embed = nn.Parameter(torch.randn(vocab, n) * cfg["input_scale"])
        self.readout = nn.Linear(n, vocab)
        nn.init.normal_(self.readout.weight, std=1.0 / math.sqrt(n))
        nn.init.zeros_(self.readout.bias)

    # ------------------------------------------------------------------------------
    def weights(self) -> torch.Tensor:
        return nn.functional.softplus(self.theta) * self.w_scale

    def initial_state(self, batch: int, warmup_ms: float = 20.0):
        """Network state after `warmup_ms` without input, for `batch` streams."""
        if self.model == "gru":
            return {"h": torch.zeros(1, batch, self.n, device=self.readout.weight.device,
                                     dtype=self.readout.weight.dtype)}
        steps = max(1, int(round(warmup_ms / self.cfg["dt"])))
        zeros = torch.zeros(batch, self.n, steps, device=self.embed.device, dtype=self.embed.dtype)
        with torch.no_grad():
            _, states = jx.integrate(
                self.bnet.net, params=[{f"{self.bnet.synapse_name}_gS": self.weights()}],
                data_stimuli=self.bnet.input_view.data_stimulate(zeros), delta_t=self.cfg["dt"],
                return_states=True,
            )
        return {k: v.detach().clone() for k, v in states.items()}

    def _token(self, states: Dict[str, torch.Tensor], current: torch.Tensor, gs: torch.Tensor):
        stim = current[..., None].expand(*current.shape, self.spt)
        v, states = jx.integrate(
            self.bnet.net, params=[{f"{self.bnet.synapse_name}_gS": gs}],
            data_stimuli=self.bnet.input_view.data_stimulate(stim), all_states=states, return_states=True,
            delta_t=self.cfg["dt"], checkpoint_lengths=self.inner_checkpoint, compile=self.compile,
        )
        feat = (v[..., 1:].mean(-1) - self.cfg["v_center"]) / self.cfg["v_scale"]
        return feat, states

    def forward(self, tokens: torch.Tensor, states: Dict[str, torch.Tensor]):
        """tokens: (batch, K). Returns logits (batch, K, vocab) and the final state."""
        if self.model == "gru":
            out, h = self.rnn(self.embed(tokens), states["h"])
            return self.readout(out), {"h": h}
        gs = self.weights()
        currents = self.embed[tokens] * self.input_nA  # (batch, K, n) in nA
        feats = []
        for k in range(tokens.shape[1]):
            if self.bptt_tokens and k and k % self.bptt_tokens == 0:
                states = {name: v.detach() for name, v in states.items()}  # Truncate the gradient.
            if torch.is_grad_enabled():
                feat, states = checkpoint(self._token, states, currents[:, k], gs, use_reentrant=False)
            else:
                feat, states = self._token(states, currents[:, k], gs)
            feats.append(feat)
        return self.readout(torch.stack(feats, dim=1)), states

    def num_params(self) -> Dict[str, int]:
        return {name: p.numel() for name, p in self.named_parameters()}
