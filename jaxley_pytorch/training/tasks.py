# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""The tasks a network is trained on, selected by `[task] name`: `cycling` (the cycling
task of Beiran & Litwin-Kumar 2025, Fig. 1) and `text` (next-token prediction on a token
file). The keys of `[task]` are the arguments of the class.

A task draws the network's input and readout weights (`init_io`) and runs the network
on its inputs (`run`). For training (`run.py`), `prepare` loads the data for the run's
seed and batch size, `set_teacher` computes a teacher's outputs once, `next_batch` and
`loss` make one training step, `evaluate` gives the task loss on the evaluation data
(and, with a teacher, how close the outputs are to the teacher's) with the rates, and
`state_dict`/`load_state_dict` hold the position in the data for checkpoints.
"""

import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from jaxley_pytorch.training import metrics
from jaxley_pytorch.training.engine import on_side_stream
from jaxley_pytorch.training.rate_network import TAU, RateNetwork, Simulator

LN2 = math.log(2.0)
FRAME = 10  # solver steps per checkpointed piece of a cycling trial


def _positive(**values):
    for key, value in values.items():
        if not (isinstance(value, int) and value > 0):
            raise ValueError(f"[task] {key} must be a positive integer, got {value!r}")


def cycling_trials(seed: int, trials: int, steps: int, dt_tau: float, omega: float):
    """The paper's cycling task (`fun_lib.create_input`), in steps of `dt_tau` tau: a
    one-step pulse on one of two input channels at a random time in [2, 5] tau, after
    which the readout must follow sin(omega (t - t_pulse)) for 2 periods (channel 0) or
    5 periods (channel 1), then 0; 20% of the trials have neither pulse nor oscillation.
    Returns inputs (trials, steps, 2) and targets (trials, steps), float32."""
    rng = np.random.default_rng(seed)
    taus = np.arange(steps) * dt_tau
    period = 2 * np.pi / omega
    inputs = np.zeros((trials, steps, 2), dtype=np.float32)
    targets = np.zeros((trials, steps), dtype=np.float32)
    for tr in range(trials):
        t_in = 2 + 3 * rng.random()
        i_in = int(np.argmin(np.abs(taus - t_in)))
        cond = int(rng.integers(2))
        inputs[tr, i_in, cond] = 1.0
        targets[tr, i_in:] = np.sin(omega * (taus[i_in:] - taus[i_in]))
        i_end = int(np.argmin(np.abs(taus - t_in - period * (2 if cond == 0 else 5))))
        targets[tr, i_end:] = 0.0
        if rng.random() < 0.2:
            inputs[tr] = 0.0
            targets[tr] = 0.0
    return inputs, targets


class Cycling:
    """`[task] name = "cycling"`: the paper's cycling task.

    `trials` fixed trials of `trial_steps` solver steps, drawn with the seed
    `trials_seed` (oscillation frequency `omega` per tau), on which every run is
    evaluated, and trained on unless `train_trials` > 0: then on that many new trials of
    the task, drawn with the seed 10,000 + the run's seed. A training step takes a batch
    of trials without replacement. Every trial starts from rest, and the readout is read
    from the initial state and after every step, as in the paper's code. The input
    weights are random (normal / alpha, so that a pulse of 1 moves x by about 1) and
    stay fixed, as in the paper; the readout starts at 5 / N * normal. Task loss: the
    mean squared error of the readout."""

    io_names = ("w_in", "w_out")

    def __init__(
        self,
        trials: int,
        trial_steps: int,
        trials_seed: int,
        omega: float,
        train_trials: int,
    ):
        _positive(trials=trials, trial_steps=trial_steps)
        if not (isinstance(train_trials, int) and train_trials >= 0):
            raise ValueError(f"[task] train_trials must be >= 0, got {train_trials!r}")
        self.trials, self.trial_steps = trials, trial_steps
        self.trials_seed, self.omega, self.train_trials = (
            trials_seed,
            omega,
            train_trials,
        )
        self.teacher = None

    def init_io(self, n: int, rng: np.random.Generator, alpha: float) -> Dict:
        return {
            "w_in": rng.standard_normal((2, n)) / alpha,
            "w_out": 5 * rng.standard_normal(n) / n,
        }

    def readout_basis(self, net: RateNetwork) -> torch.Tensor:
        """Orthonormal basis (n, 1) of the activity direction the readout reads."""
        q, _ = torch.linalg.qr(net.io["w_out"][:, None].detach().double())
        return q

    def run(self, net: RateNetwork, inputs: torch.Tensor, states: Dict):
        """Trials `inputs` (batch, steps, 2) -> readout (batch, steps), rates (batch, n,
        steps) and the final states. Step 0 reads the initial state and step t + 1 the
        state after the update with the input of step t."""
        params = net.sim_params()
        current = torch.einsum("btc,cn->bnt", inputs, net.io["w_in"])
        steps = inputs.shape[1]
        rates = [net.rates(states["v"][..., None])]
        for f in range(0, steps, FRAME):
            piece = current[..., f : f + FRAME]
            if torch.is_grad_enabled():
                v, states = checkpoint(
                    net.sim.segment, states, piece, params, use_reentrant=False
                )
            else:
                v, states = net.sim.segment(states, piece, params)
            rates.append(net.rates(v))
        a = torch.cat(rates, dim=-1)[..., :steps]
        return torch.einsum("bnt,n->bt", a, net.io["w_out"]), a, states

    def _trials(self, seed: int, trials: int, sim: Simulator):
        arrays = cycling_trials(
            seed, trials, self.trial_steps, sim.dt / TAU, self.omega
        )
        return tuple(
            torch.as_tensor(x, device=sim.device, dtype=sim.dtype) for x in arrays
        )

    def prepare(self, sim: Simulator, seed: int, batch: int):
        self.inputs, self.targets = self._trials(self.trials_seed, self.trials, sim)
        if self.train_trials:
            self.pool = self._trials(10_000 + seed, self.train_trials, sim)
        else:
            self.pool = (self.inputs, self.targets)
        if batch > len(self.pool[0]):
            raise ValueError(
                f"[trainer] batch ({batch}) is larger than the {len(self.pool[0])} "
                "training trials"
            )
        self.batch = batch
        self.static_in = torch.zeros(batch, self.trial_steps, 2, device=sim.device)
        self.static_tg = torch.zeros(batch, self.trial_steps, device=sim.device)
        self.rng = np.random.default_rng(seed + 1)

    def set_teacher(self, teacher: RateNetwork) -> torch.Tensor:
        """Compute the teacher's outputs on the evaluation trials once; returns its
        rates (trials, steps, n)."""
        z_t, a_t = self.outputs(teacher)
        loss = float(((z_t - self.targets) ** 2).mean())
        self.teacher = {"z": z_t, "task_loss": loss}
        return a_t

    def extra_baselines(self) -> Dict:
        return {}

    def next_batch(self):
        idx = self.rng.choice(len(self.pool[0]), self.batch, replace=False)
        idx = torch.as_tensor(idx, device=self.static_in.device)
        self.static_in.copy_(self.pool[0][idx])
        self.static_tg.copy_(self.pool[1][idx])

    def loss(self, net: RateNetwork):
        z, _, _ = self.run(net, self.static_in, net.sim.rest_states(self.batch))
        return ((z - self.static_tg) ** 2).mean(), None

    def carry(self, state):
        pass  # Every trial starts from rest.

    @torch.no_grad()
    def outputs(self, net: RateNetwork) -> Tuple[torch.Tensor, torch.Tensor]:
        """Readout (trials, steps) and rates (trials, steps, n) of `net` on the
        evaluation trials, in batches of the training batch (the last one padded)."""
        n_trials = self.inputs.shape[0]
        pad = (-n_trials) % self.batch
        inp = self.inputs
        if pad:
            extra = torch.arange(pad, device=inp.device) % n_trials
            inp = torch.cat([inp, inp[extra]])
        zs, rates = [], []
        for i in range(0, len(inp), self.batch):
            z, a, _ = self.run(
                net, inp[i : i + self.batch], net.sim.rest_states(self.batch)
            )
            zs.append(z)
            rates.append(a)
        return torch.cat(zs)[:n_trials], torch.cat(rates)[:n_trials].transpose(1, 2)

    def evaluate(self, net: RateNetwork) -> Tuple[Dict, torch.Tensor]:
        """The task loss of `net` on the evaluation trials (with a teacher: also the
        teacher's, and the readout's mean squared difference from the teacher's) and the
        rates (trials, steps, n)."""
        z, a = self.outputs(net)
        out = {"task_loss": float(((z - self.targets) ** 2).mean())}
        if self.teacher is not None:
            out["teacher_task_loss"] = self.teacher["task_loss"]
            out["output_vs_teacher"] = float(((z - self.teacher["z"]) ** 2).mean())
        return out, a

    def silent_loss(self) -> float:
        """The task loss of a readout that stays at zero."""
        return float((self.targets**2).mean())

    def backward_full_batch(self, net: RateNetwork) -> torch.Tensor:
        """Accumulate the gradients of the task loss on all the evaluation trials (in
        batches, the last one padded with weight 0) and return the loss."""
        n = self.inputs.shape[0]
        pad = (-n) % self.batch
        extra = torch.arange(pad, device=self.inputs.device) % n
        inp = torch.cat([self.inputs, self.inputs[extra]]) if pad else self.inputs
        tg = torch.cat([self.targets, self.targets[extra]]) if pad else self.targets
        weight = torch.cat([torch.ones(n), torch.zeros(pad)]).to(self.inputs.device)
        total = torch.zeros((), device=self.inputs.device)
        for i in range(0, len(inp), self.batch):
            rest = net.sim.rest_states(self.batch)
            z, _, _ = self.run(net, inp[i : i + self.batch], rest)
            err = ((z - tg[i : i + self.batch]) ** 2).mean(-1)
            loss = (err * weight[i : i + self.batch]).sum() / n
            loss.backward()
            total = total + loss.detach()
        return total

    def state_dict(self) -> Dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: Dict):
        self.rng.bit_generator.state = state["rng"]


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    meta.setdefault("dtype", "uint8")  # (the character-level meta.json lacks the key)
    return meta


def chars_per_token(meta: dict, split: str = "val") -> float:
    """Characters (end-of-story marks included) per token of a split."""
    info = meta["splits"][split]
    return float(info.get("chars", info["tokens"])) / info["tokens"]


class Streams:
    """`batch` parallel streams through a token file, advanced by `chunk` tokens per
    call; every call returns (batch, chunk + 1) tokens (inputs and next-token
    targets)."""

    def __init__(self, path: str, dtype: str, batch: int, chunk: int, seed: int = 0):
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.pos = np.random.default_rng(seed).integers(0, len(self.data), size=batch)
        self.chunk = chunk

    def next(self) -> np.ndarray:
        idx = (self.pos[:, None] + np.arange(self.chunk + 1)) % len(self.data)
        self.pos = (self.pos + self.chunk) % len(self.data)
        return np.asarray(self.data[idx], dtype=np.int64)


def eval_segments(path: str, dtype: str, n_segments: int, length: int) -> np.ndarray:
    """`n_segments` windows of `length + 1` tokens spread evenly over a token file."""
    data = np.memmap(path, dtype=dtype, mode="r")
    starts = np.linspace(0, len(data) - length - 2, n_segments).astype(np.int64)
    return np.stack(
        [np.asarray(data[s : s + length + 1], dtype=np.int64) for s in starts]
    )


class Text:
    """`[task] name = "text"`: next-token prediction on the token files in `data`
    (`train.bin`, `val.bin` and `meta.json`, as written by
    examples/language_modeling/prepare_data.py and
    examples/activitome-connectome/task-complexity/prepare_tokens.py).

    Every token is a constant input current for `steps_per_token` solver steps,
    `embed_proj @ embed[token]` with a `width`-wide embedding; the logits are `unembed @
    (read_proj @ a) + read_bias`, with `a` the network's activity (`features`: a rate
    network's rates) averaged over the token's window. Training: as many streams through
    `train.bin` as the batch, `chunk` tokens per step (truncated backpropagation through
    time; the state carries over from step to step; with `bptt_tokens`, the gradient is
    also cut every `bptt_tokens` tokens inside a chunk, the state still carried over),
    cross-entropy. Evaluation:
    `eval_segments` segments spread over `val.bin`, each from rest, `eval_burn` tokens
    and then `eval_length` measured ones; task loss in bits per token. For a spiking
    network also its firing rate and, with a teacher, how close its spike counts per
    token are to the teacher's (next to the teacher's own against a permutation of its
    neurons, `extra_baselines`)."""

    io_names = ("embed", "embed_proj", "read_proj", "unembed", "read_bias")

    def __init__(
        self,
        data: str,
        width: int,
        steps_per_token: int,
        chunk: int,
        eval_segments: int,
        eval_burn: int,
        eval_length: int,
        bptt_tokens: Optional[int] = None,
    ):
        _positive(
            width=width,
            steps_per_token=steps_per_token,
            chunk=chunk,
            eval_segments=eval_segments,
            eval_length=eval_length,
        )
        if not (isinstance(eval_burn, int) and eval_burn >= 0):
            raise ValueError(f"[task] eval_burn must be >= 0, got {eval_burn!r}")
        if bptt_tokens is not None and not (
            isinstance(bptt_tokens, int) and 1 <= bptt_tokens <= chunk
        ):
            raise ValueError(
                f"[task] bptt_tokens must be an integer from 1 to chunk ({chunk}), "
                f"got {bptt_tokens!r}"
            )
        self.bptt_tokens = bptt_tokens
        self.data, self.width, self.steps_per_token = data, width, steps_per_token
        self.chunk, self.eval_segments = chunk, eval_segments
        self.eval_burn, self.eval_length = eval_burn, eval_length
        self.meta = load_meta(data)
        self.vocab = int(self.meta["vocab_size"])
        self.chars_per_token = chars_per_token(self.meta)
        self.teacher = None

    def init_io(self, n: int, rng: np.random.Generator, alpha: float) -> Dict:
        k = self.width
        return {
            "embed": rng.standard_normal((self.vocab, k)),
            "embed_proj": rng.standard_normal((n, k)) / math.sqrt(k),
            "read_proj": rng.standard_normal((k, n)) / math.sqrt(n),
            "unembed": rng.standard_normal((self.vocab, k)) / math.sqrt(k),
            "read_bias": np.zeros(self.vocab),
        }

    def readout_basis(self, net: RateNetwork) -> torch.Tensor:
        """Orthonormal basis (n, width) of the activity directions the readout reads."""
        q, _ = torch.linalg.qr(net.io["read_proj"].T.detach().double())
        return q

    def _simulate(self, net, states: Dict, current: torch.Tensor, params):
        stim = current[..., None].expand(*current.shape, self.steps_per_token)
        return net.sim.segment(states, stim, params)

    def _token(self, net, states: Dict, current: torch.Tensor, params):
        v, states = self._simulate(net, states, current, params)
        return net.features(v).mean(-1), states

    def run(self, net, tokens: torch.Tensor, states: Dict, spikes: bool = False):
        """`tokens` (batch, K) -> logits (batch, K, vocab), window-mean activity
        (batch, K, n) and the final states, and with `spikes` (without gradients, a
        spiking network) the spike counts (batch, K, n). Every token's simulation is
        checkpointed; with `bptt_tokens`, the gradient through the states is cut every
        `bptt_tokens` tokens."""
        params = net.sim_params()
        currents = net.io["embed"][tokens] @ net.io["embed_proj"].T
        feats, counts = [], []
        for k in range(tokens.shape[1]):
            if torch.is_grad_enabled():
                if self.bptt_tokens and k and k % self.bptt_tokens == 0:
                    states = {name: s.detach() for name, s in states.items()}
                a, states = checkpoint(
                    self._token,
                    net,
                    states,
                    currents[:, k],
                    params,
                    use_reentrant=False,
                )
            elif spikes:
                v, after = self._simulate(net, states, currents[:, k], params)
                a = net.features(v).mean(-1)
                counts.append(net.spike_counts(states["v"], v))
                states = after
            else:
                a, states = self._token(net, states, currents[:, k], params)
            feats.append(a)
        feats = torch.stack(feats, dim=1)
        readout = feats @ net.io["read_proj"].T
        logits = readout @ net.io["unembed"].T + net.io["read_bias"]
        if spikes:
            return logits, feats, states, torch.stack(counts, dim=1)
        return logits, feats, states

    def prepare(self, sim: Simulator, seed: int, batch: int):
        if self.eval_segments % batch:
            raise ValueError(
                f"[task] eval_segments ({self.eval_segments}) must be a multiple of "
                f"[trainer] batch ({batch})"
            )
        dtype = self.meta["dtype"]
        self.batch = batch
        train = os.path.join(self.data, "train.bin")
        self.streams = Streams(train, dtype, batch, self.chunk, seed=seed)
        length = self.eval_burn + self.eval_length
        segs = eval_segments(
            os.path.join(self.data, "val.bin"), dtype, self.eval_segments, length
        )
        self.eval_tok = torch.as_tensor(segs, device=sim.device)
        self.static_tok = torch.zeros(
            batch, self.chunk + 1, dtype=torch.long, device=sim.device
        )
        self.static_states = on_side_stream(lambda: sim.rest_states(batch))

    def set_teacher(self, teacher) -> torch.Tensor:
        """Remember the teacher (its next-token distributions are compared with the
        student's at every evaluation); returns its activity (segments, tokens, n)."""
        out = self._outputs(teacher, None)
        self.teacher = teacher
        self.spike_baselines = (
            metrics.spike_baselines(out["c_s"]) if "c_s" in out else {}
        )
        return out["a_s"]

    def extra_baselines(self) -> Dict:
        """The shuffled baselines of the spike counts (a spiking teacher)."""
        return dict(getattr(self, "spike_baselines", {}))

    def next_batch(self):
        self.static_tok.copy_(torch.as_tensor(self.streams.next()))

    def loss(self, net: RateNetwork):
        logits, _, state = self.run(net, self.static_tok[:, :-1], self.static_states)
        target = self.static_tok[:, 1:].reshape(-1)
        return F.cross_entropy(logits.reshape(-1, self.vocab), target), state

    def carry(self, state: Dict):
        with torch.no_grad():
            for k in self.static_states:
                self.static_states[k].copy_(state[k])

    @torch.no_grad()
    def _outputs(self, net, teacher) -> Dict:
        """Window-mean activity (segments, tokens after the burn-in, n) of `net` (and
        `teacher`) on the evaluation segments, every segment from rest, the summed
        cross-entropies (and KL divergence from the teacher) of the measured tokens
        (nats), and the spike counts of a spiking network (same shape)."""
        burn, tokens = self.eval_burn, self.eval_tok
        out = {"ce_s": 0.0, "ce_t": 0.0, "kl": 0.0, "count": 0}
        acts, counts = {"s": [], "t": []}, {"s": [], "t": []}
        for i in range(0, tokens.shape[0], self.batch):
            tok = tokens[i : i + self.batch]
            target = tok[:, 1 + burn :].reshape(-1)
            logp = {}
            for name, model in (("s", net), ("t", teacher)):
                if model is None:
                    continue
                rest = model.sim.rest_states(tok.shape[0])
                spikes = hasattr(model, "spike_counts")
                result = self.run(model, tok[:, :-1], rest, spikes=spikes)
                logits, feats = result[0], result[1]
                if spikes:
                    counts[name].append(result[3][:, burn:])
                logp[name] = F.log_softmax(logits[:, burn:].float(), dim=-1)
                flat = logp[name].reshape(-1, logits.shape[-1])
                out[f"ce_{name}"] += F.nll_loss(flat, target, reduction="sum").item()
                acts[name].append(feats[:, burn:])
            if teacher is not None:
                out["kl"] += (logp["t"].exp() * (logp["t"] - logp["s"])).sum().item()
            out["count"] += target.numel()
        for name in ("s", "t"):
            if acts[name]:
                out[f"a_{name}"] = torch.cat(acts[name])
            if counts[name]:
                out[f"c_{name}"] = torch.cat(counts[name])
        return out

    def evaluate(self, net: RateNetwork) -> Tuple[Dict, torch.Tensor]:
        """The task loss of `net` on the evaluation segments (bits per token and per
        character; with a teacher: also the teacher's, and the KL divergence of the
        next-token distributions from the teacher's, bits per token; a spiking
        network's firing rate, and with a teacher its spike counts against the
        teacher's, `metrics.spike_compare`) and the activity (segments, tokens, n)."""
        ev = self._outputs(net, self.teacher)
        n = ev["count"]
        out = {"task_loss": ev["ce_s"] / n / LN2}
        if self.teacher is not None:
            out["teacher_task_loss"] = ev["ce_t"] / n / LN2
        out["val_bits_per_char"] = out["task_loss"] / self.chars_per_token
        if self.teacher is not None:
            out["output_vs_teacher"] = ev["kl"] / n / LN2
        if "c_s" in ev:
            token_s = self.steps_per_token * net.sim.dt / 1000.0
            out["firing_rate_hz"] = float(ev["c_s"].mean()) / token_s
            if "c_t" in ev:
                out["teacher_firing_rate_hz"] = float(ev["c_t"].mean()) / token_s
                out.update(metrics.spike_compare(ev["c_s"], ev["c_t"]))
        return out, ev["a_s"]

    def state_dict(self) -> Dict:
        return {
            "pos": torch.as_tensor(self.streams.pos),
            "states": {k: v.clone() for k, v in self.static_states.items()},
        }

    def load_state_dict(self, state: Dict):
        self.streams.pos = state["pos"].cpu().numpy()
        for k, v in state["states"].items():
            self.static_states[k].copy_(v)


TASKS = {"cycling": Cycling, "text": Text}
