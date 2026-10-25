"""The tasks: the paper's cycling task, and next-token prediction on TinyStories at the
level of characters (`chars`) or of BPE tokens (`tokens`)."""
import json
import os

import numpy as np

DATA = {
    # The character-level files of examples/language_modeling (74 ASCII characters and an
    # end-of-story token: every byte that occurs in the dataset).
    "chars": "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/language_modeling/data",
    # prepare_tokens.py (byte-level BPE, 4,096 tokens).
    "tokens": "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/task-complexity/data/bpe4096",
}
TRIALS = 60
TRIAL_STEPS = 400  # 20 tau in steps of 0.05 tau


def cycling_trials(seed: int = 21, trials: int = TRIALS, steps: int = TRIAL_STEPS, dt_tau: float = 0.05,
                   omega: float = 3.0):
    """The paper's cycling task (`fun_lib.create_input`): a one-step pulse on one of two
    input channels at a random time in [2, 5] tau, after which the readout must follow
    sin(omega (t - t_pulse)) for 2 periods (channel 0) or 5 periods (channel 1), then 0;
    20% of the trials have neither pulse nor oscillation. Returns inputs (trials, steps, 2)
    and targets (trials, steps), float32."""
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


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    meta.setdefault("dtype", "uint8")  # (the character-level meta.json predates the key)
    return meta


def chars_per_token(meta: dict, split: str = "val") -> float:
    """Characters (end-of-story marks included) per token of a split."""
    info = meta["splits"][split]
    return float(info.get("chars", info["tokens"])) / info["tokens"]


class Streams:
    """`batch` parallel streams through a token file, advanced by `chunk` tokens per
    call; every call returns (batch, chunk + 1) tokens (inputs and next-token targets)."""

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
    return np.stack([np.asarray(data[s : s + length + 1], dtype=np.int64) for s in starts])
