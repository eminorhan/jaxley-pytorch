# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""`jaxley_pytorch.training.config`: what a config file must and may contain."""

import dataclasses
import re

import pytest
import torch

from jaxley_pytorch.training.config import load_config
from jaxley_pytorch.training.engine import lr_factor, make_optimizer
from jaxley_pytorch.training.rate_network import BLKRate

HEAD = """
experiment_name = "arm"
seeds = [0]
"""
MODEL = """
[model]
name = "blk_rate"
neurons = 20
exc_frac = 0.7
connection_prob = 0.5
dt = 0.5
"""
INIT = """
[init]
teacher = "teacher/seed0"
start = "shuffled"
"""
CYCLING = """
[task]
name = "cycling"
trials = 8
trial_steps = 40
trials_seed = 21
omega = 3.0
train_trials = 0
"""
TEXT = """
[task]
name = "text"
data = "/no/such/dir"
width = 4
steps_per_token = 2
chunk = 4
eval_segments = 4
eval_burn = 2
eval_length = 4
"""
TRAINER = """
[trainer]
trainable = ["W", "g", "b", "w_out"]
steps = 3
warmup_steps = 1
batch = 4
lr = 1e-3
clip = 5.0
clip_mode = "global"
eval_every = 2
log_every = 1
dtype = "float64"
"""
TEACHER = HEAD + MODEL + CYCLING + TRAINER


def load(tmp_path, text):
    path = tmp_path / "arm.toml"
    path.write_text(text)
    return load_config(path)


def test_valid(tmp_path):
    cfg = load(tmp_path, TEACHER)
    assert cfg.model.name == "blk_rate" and cfg.model.kwargs["neurons"] == 20
    assert cfg.init is None and cfg.seeds == (0,)
    assert cfg.trainer.trainable == ("W", "g", "b", "w_out")
    assert cfg.trainer.checkpoint_every == 0 and cfg.trainer.cuda_graph  # defaults
    assert cfg.trainer.optimizer == "adamw" and cfg.trainer.weight_decay == 0.0
    student = load(tmp_path, HEAD + INIT + CYCLING + TRAINER)
    assert student.model is None and student.init.start == "shuffled"


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("[trainer]\n", "[trainer]\nmomentum = 0.9\n", "unknown key(s) ['momentum']"),
        ("omega = 3.0\n", "", "missing ['omega']"),
        ("seeds = [0]\n", "seeds = [0]\nnotes = 1\n", "unknown top-level"),
        ('name = "cycling"', 'name = "cyclng"', "name must be one of"),
        ('"W", "g", "b", "w_out"', '"W", "gain"', "unknown parameter(s) ['gain']"),
        ('["W", "g", "b", "w_out"]', '"g"', "a list of parameter names"),
        ('clip_mode = "global"', 'clip_mode = "per-param"', "clip_mode must be one of"),
        ("seeds = [0]", "seeds = [1, 1]", "distinct"),
        ("warmup_steps = 1", "warmup_steps = 5", "warmup_steps must be at most steps"),
        ('experiment_name = "arm"', 'experiment_name = "a/b"', "a directory name"),
        ("dtype =", 'optimizer = "sgd"\ndtype =', "optimizer must be one of"),
        ("dtype =", 'optimizer = "adam"\nweight_decay = 0.1\ndtype =', "goes with"),
    ],
)
def test_invalid(tmp_path, old, new, message):
    assert old in TEACHER
    with pytest.raises(ValueError, match=re.escape(message)):
        load(tmp_path, TEACHER.replace(old, new, 1))


def test_model_or_init(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        load(tmp_path, HEAD + MODEL + INIT + CYCLING + TRAINER)
    with pytest.raises(ValueError, match="not both"):
        load(tmp_path, HEAD + CYCLING + TRAINER)
    bad = INIT.replace('start = "shuffled"', 'start = "shuffled"\nstable_start = true')
    with pytest.raises(ValueError, match="stable_start"):
        load(tmp_path, HEAD + bad + CYCLING + TRAINER)


def test_fan_in(tmp_path):
    """`fan_in` instead of `connection_prob`, and exactly one of the two."""
    fan_in = TEACHER.replace("connection_prob = 0.5", "fan_in = 10")
    model = BLKRate(**load(tmp_path, fan_in).model.kwargs)
    assert model.p == 0.5 and model.weight_n == 20
    both = TEACHER.replace(
        "connection_prob = 0.5", "connection_prob = 0.5\nfan_in = 10"
    )
    neither = TEACHER.replace("connection_prob = 0.5\n", "")
    for text in (both, neither):
        with pytest.raises(
            ValueError, match="exactly one of connection_prob and fan_in"
        ):
            BLKRate(**load(tmp_path, text).model.kwargs)


def test_spiking_model(tmp_path):
    """A spiking model's keys are all required; its networks are for the text task."""
    spiking = (
        '[model]\nname = "hh"\nneurons = 20\nexc_frac = 0.8\nfan_in = 5\ndt = 0.025\n'
        "input_scale = 6.0\nw_exc = 0.02\nw_inh = 0.1\nspread = 0.25\n"
    )
    trainer = TRAINER.replace('"W", "g", "b", "w_out"', '"W", "gNa", "bias"')
    cfg = load(tmp_path, HEAD + spiking + TEXT + trainer)
    assert cfg.model.name == "hh" and cfg.model.kwargs["fan_in"] == 5
    with pytest.raises(ValueError, match=re.escape("missing ['spread']")):
        load(tmp_path, HEAD + spiking.replace("spread = 0.25\n", "") + TEXT + trainer)
    with pytest.raises(ValueError, match=re.escape("unknown parameter(s) ['g']")):
        load(tmp_path, HEAD + spiking + TEXT + trainer.replace('"gNa"', '"g"'))
    with pytest.raises(ValueError, match="spiking neurons are for the text task"):
        load(tmp_path, HEAD + spiking + CYCLING + trainer.replace('"bias"', '"w_out"'))


def test_text_rules(tmp_path):
    trainer = TRAINER.replace('"W", "g", "b", "w_out"', '"g", "b"')
    assert load(tmp_path, HEAD + MODEL + TEXT + trainer).task.name == "text"
    with pytest.raises(ValueError, match="lbfgs_steps is for the cycling task"):
        load(tmp_path, HEAD + MODEL + TEXT + trainer + "lbfgs_steps = 2\n")
    with pytest.raises(ValueError, match="multiple of"):
        load(tmp_path, HEAD + MODEL + TEXT + trainer.replace("batch = 4", "batch = 3"))


def test_weight_decay_groups(tmp_path):
    """AdamW decays the weights, not the gains, biases and readout bias; Adam is plain."""
    cfg = dataclasses.replace(load(tmp_path, TEACHER).trainer, weight_decay=0.05)
    names = ("W", "g", "b", "embed", "read_bias")
    named = [(n, torch.nn.Parameter(torch.ones(3))) for n in names]
    opt, _ = make_optimizer(named, cfg, use_graph=False)
    assert type(opt) is torch.optim.AdamW
    decay = {g["weight_decay"]: {id(p) for p in g["params"]} for g in opt.param_groups}
    assert decay == {
        0.05: {id(p) for n, p in named if n in ("W", "embed")},
        0.0: {id(p) for n, p in named if n in ("g", "b", "read_bias")},
    }
    adam = dataclasses.replace(cfg, optimizer="adam", weight_decay=0.0)
    opt, _ = make_optimizer(named, adam, use_graph=False)
    assert type(opt) is torch.optim.Adam and len(opt.param_groups) == 1


def test_lr_factor():
    f = lr_factor(10, 4)
    assert [f(s) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert f(4) == 1.0 and f(9) == pytest.approx(1 / 6) and f(10) == 0.0
