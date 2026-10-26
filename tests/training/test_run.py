# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""`python -m jaxley_pytorch.training` end to end, on the CPU with small networks: a
teacher and students of either task, the run directory, refusing to overwrite a run, and
resuming one from its checkpoint."""

import json

import numpy as np
import pytest
import torch

from jaxley_pytorch.training.run import main

MODEL = """
[model]
name = "blk_rate"
neurons = 32
exc_frac = 0.7
connection_prob = 0.5
dt = 0.5
"""
CYCLING = """
[task]
name = "cycling"
trials = 6
trial_steps = 120
trials_seed = 21
omega = 3.0
train_trials = {train_trials}
"""
TEXT = """
[task]
name = "text"
data = "{data}"
width = 4
steps_per_token = 2
chunk = 4
eval_segments = 4
eval_burn = 2
eval_length = 4
"""
TRAINER = """
[trainer]
trainable = {trainable}
steps = {steps}
warmup_steps = 1
batch = {batch}
lr = 1e-3
clip = 2.0
clip_mode = "global"
eval_every = 2
log_every = 1
dtype = "float64"
compile = false
cuda_graph = false
"""
TEACHER = '["W", "g", "b", "w_out"]'
TEXT_TEACHER = '["W", "g", "b", "embed", "embed_proj", "read_proj", "unembed", "read_bias"]'


def head(name, seeds=(0,)):
    return f'experiment_name = "{name}"\nseeds = {list(seeds)}\n'


def init(start="shuffled", extra=""):
    return f'[init]\nteacher = "teacher/seed0"\nstart = "{start}"\n{extra}'


def trainer(trainable, steps=3, batch=3, extra=""):
    return TRAINER.format(trainable=trainable, steps=steps, batch=batch) + extra


def run(tmp_path, name, text, *args):
    path = tmp_path / f"{name}.toml"
    path.write_text(text)
    root = tmp_path / "runs"
    return main(["--config", str(path), "--output-root", str(root), "--device", "cpu", *args])


def result(out):
    return json.loads((out / "result.json").read_text())


def test_cycling(tmp_path):
    cycling = CYCLING.format(train_trials=0)
    out = run(tmp_path, "teacher", head("teacher") + MODEL + cycling + trainer(TEACHER))
    assert out == tmp_path / "runs" / "teacher" / "seed0"
    for name in ["config.toml", "resolved_config.json", "git_commit.txt", "network.pt"]:
        assert (out / name).exists()
    assert not (out / "checkpoint.pt").exists()
    r = result(out)
    assert [e["step"] for e in r["evals"]] == [0, 2, 3] and r["final"]["step"] == 3
    assert r["neurons"] == 32 and "participation_ratio" in r["final"]

    students = head("students", (1, 2)) + init() + cycling + trainer('["g", "b"]', steps=2)
    out = run(tmp_path, "students", students, "--seed", "2")
    r = result(out)
    for key in ["act_mse", "act_corr_err", "teacher_task_loss", "output_vs_teacher"]:
        assert np.isfinite(r["final"][key])
    assert r["baselines"]["act_mse_shuffled"] > 0
    resolved = json.loads((out / "resolved_config.json").read_text())
    assert resolved["seed"] == 2 and resolved["model"]["neurons"] == 32
    assert resolved["init"]["teacher"] == str(tmp_path / "runs" / "teacher" / "seed0")

    # Another student's gains and biases, then full-batch L-BFGS only.
    text = head("continued", (2,)) + init("student", 'student = "students/seed{seed}"\n')
    lbfgs = "lbfgs_steps = 2\nlbfgs_eval_every = 1\n"
    text += cycling + trainer('["g", "b"]', steps=0, extra=lbfgs)
    c = result(run(tmp_path, "continued", text))
    assert c["evals"][0]["gain_rmse"] == pytest.approx(r["final"]["gain_rmse"])
    assert [e.get("phase") for e in c["evals"]] == [None, "lbfgs", "lbfgs"]

    # A random start drawn again while it runs away, on new trials of the task.
    text = head("rstable", (3,)) + init("random", "stable_start = true\n")
    text += CYCLING.format(train_trials=12) + trainer('["g", "b"]', steps=2)
    r = result(run(tmp_path, "rstable", text))
    assert r["start"]["redraws"] >= 0 and np.isfinite(r["final"]["task_loss"])


@pytest.fixture
def text_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 12, 3000).astype(np.uint8).tofile(d / "train.bin")
    rng.integers(0, 12, 600).astype(np.uint8).tofile(d / "val.bin")
    meta = {"vocab_size": 12, "dtype": "uint8", "splits": {"val": {"tokens": 600}}}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_text(tmp_path, text_data):
    text = TEXT.format(data=text_data)
    teacher = head("teacher") + MODEL + text + trainer(TEXT_TEACHER, batch=2)
    r = result(run(tmp_path, "teacher", teacher))
    assert np.isfinite(r["final"]["val_bits_per_char"])

    students = head("students", (1,)) + init() + text + trainer('["g", "b"]', batch=2)
    r = result(run(tmp_path, "students", students))
    assert r["final"]["output_vs_teacher"] >= 0 and r["final"]["act_mse_readout"] >= 0

    other = text.replace(str(text_data), str(tmp_path))
    with pytest.raises(SystemExit, match="the teacher was trained with"):
        run(tmp_path, "other", head("other") + init() + other + trainer('["g", "b"]', batch=2))


def test_refuse_and_resume(tmp_path):
    cycling = CYCLING.format(train_trials=0)
    body = MODEL + cycling + trainer(TEACHER, steps=4, extra="checkpoint_every = 2\n")
    full = run(tmp_path, "full", head("full") + body)
    with pytest.raises(SystemExit, match="already holds a run"):
        run(tmp_path, "full", head("full") + body)
    assert run(tmp_path, "full", head("full") + body, "--resume") == full  # finished

    # Stopped by the time limit after one step, then resumed: the same network as the
    # run that was not interrupted.
    out = run(tmp_path, "stopped", head("stopped") + body + "hours = 1e-9\n")
    assert (out / "checkpoint.pt").exists() and result(out)["timing"]["steps"] == 1
    run(tmp_path, "stopped", head("stopped") + body, "--resume")
    assert not (out / "checkpoint.pt").exists()
    a = torch.load(full / "network.pt", weights_only=True)["state"]
    b = torch.load(out / "network.pt", weights_only=True)["state"]
    for key in a:
        assert torch.equal(a[key], b[key]), key
    assert result(out)["final"]["task_loss"] == result(full)["final"]["task_loss"]
