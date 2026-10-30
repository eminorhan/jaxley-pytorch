# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""One arm of an experiment, for one seed:

    python -m jaxley_pytorch.training --config <arm>.toml --output-root <dir>
        [--seed k] [--resume] [--smoke]

(`config.py` describes the file). The run directory, <output root>/<experiment_name>/
seed<k>, gets:

  config.toml           the config file as written;
  resolved_config.json  every value the run uses: the seed, the absolute paths, and
                        for a run from a teacher the teacher's [model];
  git_commit.txt        the commit of the code, with "(dirty)" and dirty.patch when it
                        had uncommitted changes (code run from a snapshot of the
                        repository without .git: the record written with the snapshot);
  attempts.log          one line per start or resume;
  result.json           the log: training loss, evaluations, timing; written at every
                        evaluation, complete when it has "final";
  network.pt            (a run from scratch) the connectivity and every parameter;
  student.pt            (a run from a teacher) its gains and biases, and any other
                        parameter it trained;
  checkpoint.pt         while the run is unfinished, its last checkpoint.

A directory that already holds a run is refused unless `--resume`: that continues the
run from its checkpoint with the same settings (only the time limit may change), or
starts it again if it has none. `--smoke` makes a quick check of a config: at most 20
steps, an evaluation every 10, the loss logged every 5, at most 2 L-BFGS steps (use a
separate output root).
"""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

import jaxley_pytorch
from jaxley_pytorch.training import engine, init, metrics, networks, rate_network
from jaxley_pytorch.training.config import Component, RunConfig, load_config
from jaxley_pytorch.training.tasks import TASKS

REPO = Path(jaxley_pytorch.__file__).resolve().parent.parent
# What a student's [task] must share with its teacher's (the rest may differ).
SAME_AS_TEACHER = {"cycling": (), "text": ("data", "width", "steps_per_token")}


def pick_seed(cfg: RunConfig, seed: Optional[int]) -> int:
    if seed is None:
        if len(cfg.seeds) > 1:
            raise SystemExit(
                f"the config has seeds {list(cfg.seeds)}: pick one with --seed"
            )
        return cfg.seeds[0]
    if seed not in cfg.seeds:
        raise SystemExit(
            f"--seed {seed} is not one of the config's seeds {list(cfg.seeds)}"
        )
    return seed


def smoke(cfg: RunConfig) -> RunConfig:
    t = cfg.trainer
    trainer = dataclasses.replace(
        t,
        steps=min(t.steps, 20),
        warmup_steps=1,
        eval_every=10,
        log_every=5,
        checkpoint_every=10 if t.checkpoint_every else 0,
        hours=0.0,
        lbfgs_steps=min(t.lbfgs_steps, 2),
        lbfgs_eval_every=1,
    )
    return dataclasses.replace(cfg, trainer=trainer)


def run_path(root: Path, path: str, seed: int) -> Path:
    """A run directory named in a config: relative to the output root unless absolute;
    `{seed}` stands for the run's seed."""
    p = Path(path.replace("{seed}", str(seed)))
    return p if p.is_absolute() else root / p


def check_task(teacher_task: Dict, task: Component):
    if teacher_task["name"] != task.name:
        raise SystemExit(
            f"[task] name is {task.name!r}, but the teacher was trained on "
            f"{teacher_task['name']!r}"
        )
    for key in SAME_AS_TEACHER[task.name]:
        if key in teacher_task:
            mine, theirs = task.kwargs[key], teacher_task[key]
            if key == "data":
                mine, theirs = os.path.normpath(mine), os.path.normpath(theirs)
            if mine != theirs:
                raise SystemExit(
                    f"[task] {key} is {task.kwargs[key]!r}, but the teacher was "
                    f"trained with {teacher_task[key]!r}"
                )


def resolve(cfg: RunConfig, seed: int, root: Path, smoke_run: bool) -> Dict:
    """Every value the run uses."""
    out = {
        "experiment_name": cfg.experiment_name,
        "seed": seed,
        "seeds": list(cfg.seeds),
    }
    out["smoke"] = smoke_run
    if cfg.init is None:
        out["model"] = cfg.model.settings()
        out["init"] = None
    else:
        teacher = run_path(root, cfg.init.teacher, seed)
        settings = rate_network.network_settings(str(teacher))
        check_task(settings["task"], cfg.task)
        out["model"] = dict(settings["model"], from_teacher=True)
        student = run_path(root, cfg.init.student, seed) if cfg.init.student else ""
        out["init"] = dict(
            dataclasses.asdict(cfg.init), teacher=str(teacher), student=str(student)
        )
    out["task"] = cfg.task.settings()
    out["trainer"] = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in dataclasses.asdict(cfg.trainer).items()
    }
    return out


def _flat(d: Dict, prefix: str = "") -> Dict:
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flat(v, f"{prefix}{k}."))
        else:
            flat[f"{prefix}{k}"] = v
    return flat


def _git(*args: str, ok_codes=(0,)) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return r.stdout if r.returncode in ok_codes else None


def code_version() -> Dict:
    """The commit of the code that runs, and its uncommitted changes as a patch
    (untracked files included, except files over 1 MB)."""
    if (REPO / ".git").exists():
        commit = _git("rev-parse", "HEAD")
        if commit is not None:
            patch = _git("diff", "HEAD") or ""
            untracked = _git("ls-files", "--others", "--exclude-standard") or ""
            for name in untracked.splitlines():
                if (REPO / name).stat().st_size > 1e6:
                    patch += f"# untracked, over 1 MB, not included: {name}\n"
                    continue
                # `git diff --no-index` exits with 1 when the files differ.
                diff = _git(
                    "diff", "--no-index", "--", "/dev/null", name, ok_codes=(0, 1)
                )
                patch += diff or ""
            dirty = " (dirty)" if patch.strip() else ""
            return {"stamp": commit.strip() + dirty, "patch": patch}
    # A snapshot of the repository (probes/jaxley_pytorch/tools/snapshot.sh on /nrs).
    record = REPO / "git_commit.txt"
    if record.exists():
        patch = REPO / "dirty.patch"
        return {
            "stamp": record.read_text().strip(),
            "patch": patch.read_text() if patch.exists() else "",
        }
    return {
        "stamp": f"unknown (no git repository or snapshot record at {REPO})",
        "patch": "",
    }


def prepare_run_dir(out: Path, config_path: Path, resolved: Dict, resume: bool):
    record = out / "resolved_config.json"
    if record.exists():
        if not resume:
            raise SystemExit(
                f"{out} already holds a run: pass --resume to continue it, or change "
                "experiment_name"
            )
        before, after = _flat(json.loads(record.read_text())), _flat(resolved)
        changed = [
            k
            for k in sorted(set(before) | set(after))
            if k != "trainer.hours" and before.get(k) != after.get(k)
        ]
        if changed:
            raise SystemExit(f"cannot resume {out}: changed settings {changed}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "config.toml")
    record.write_text(json.dumps(resolved, indent=1))
    version = code_version()
    (out / "git_commit.txt").write_text(version["stamp"] + "\n")
    if version["patch"].strip():
        (out / "dirty.patch").write_text(version["patch"])
    elif (out / "dirty.patch").exists():
        os.remove(out / "dirty.patch")  # from an earlier attempt
    with open(out / "attempts.log", "a") as f:
        now = datetime.now().isoformat(timespec="seconds")
        f.write(f"{now}  {'resume' if resume else 'start'}  {version['stamp']}\n")


def finished(out: Path) -> bool:
    path = out / "result.json"
    if not path.exists() or (out / "checkpoint.pt").exists():
        return False
    with open(path) as f:
        return "final" in json.load(f)


def train(
    cfg: RunConfig, seed: int, out: Path, root: Path, device: torch.device
) -> Dict:
    """Build the network (from scratch, or from the teacher), train it, and save it;
    returns the log (also written to result.json)."""
    t = cfg.trainer
    dtype = getattr(torch, t.dtype)
    torch.set_default_dtype(dtype)
    task = TASKS[cfg.task.name](**cfg.task.kwargs)
    teacher, start = None, {}
    if cfg.init is None:
        model = networks.MODELS[cfg.model.name](**cfg.model.kwargs)
        net, conn = networks.new_network(
            model, task.init_io, seed, device, dtype, t.compile
        )
        settings = {"model": model.settings(), "task": cfg.task.settings()}
        task.prepare(net.sim, seed, t.batch)
    else:
        teacher_dir = run_path(root, cfg.init.teacher, seed)
        teacher, conn, settings = networks.load_network(
            str(teacher_dir), device, dtype, t.compile
        )
        task.prepare(teacher.sim, seed, t.batch)
        a_t = task.set_teacher(teacher)
        if cfg.init.stable_start:
            net, start = init.stable_student(
                teacher,
                conn,
                cfg.init.start,
                seed,
                lambda candidate: task.evaluate(candidate)[0]["task_loss"],
                task.silent_loss(),
            )
            print(json.dumps({"start": start}), flush=True)
        else:
            student = cfg.init.student and str(run_path(root, cfg.init.student, seed))
            net = init.make_student(teacher, conn, cfg.init.start, seed, student)
    params = net.set_trainable(t.trainable)
    named = [(n, p) for n, p in net.named_weights() if p.requires_grad]

    if teacher is None:

        def evaluate(step):
            ev, a = task.evaluate(net)
            return {
                "step": step,
                **ev,
                **metrics.activity_stats(a, task.readout_basis(net)),
            }

    else:
        basis = task.readout_basis(teacher)

        def evaluate(step):
            ev, a = task.evaluate(net)
            return {"step": step, **ev, **metrics.compare(a, a_t, net, teacher, basis)}

    log = {
        "experiment_name": cfg.experiment_name,
        "seed": seed,
        "task": cfg.task.name,
        "neurons": int(conn["n"]),
        "synapses": int(net.sim.real.sum()),
        "k": net.sim.k,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "trainable": {n: p.numel() for n, p in net.named_weights() if p.requires_grad},
        "train": [],
        "evals": [],
    }
    if teacher is not None:
        log["start"] = start
        log["teacher"] = {"dir": str(teacher_dir), **metrics.activity_stats(a_t, basis)}
        log["baselines"] = {
            **metrics.baselines(a_t, teacher, basis),
            **task.extra_baselines(),
        }
        print(
            json.dumps({"teacher": log["teacher"], "baselines": log["baselines"]}),
            flush=True,
        )

    def save_log():
        with open(out / "result.json", "w") as f:
            json.dump(log, f, indent=1)

    def get_state():
        return {"net": net.state_dict(), "task": task.state_dict()}

    def set_state(state):
        net.load_state_dict(state["net"])
        task.load_state_dict(state["task"])

    def save_network():
        if teacher is None:
            rate_network.save_network(
                str(out / "network.pt"), net, conn, settings["model"], settings["task"]
            )
        else:
            # The single-neuron parameters and everything trained.
            keep = set(net.neuron_param_names)
            st = {
                n: p.detach().cpu()
                for n, p in net.named_weights()
                if n in keep or p.requires_grad
            }
            torch.save(dict(st, seed=seed, start=cfg.init.start), out / "student.pt")

    checkpoint = engine.Checkpoint(str(out / "checkpoint.pt"), get_state, set_state)
    done = engine.fit(
        t,
        named,
        lambda: task.loss(net),
        task.carry,
        task.next_batch,
        evaluate,
        log,
        save_log,
        checkpoint,
    )
    save_network()
    if done and t.lbfgs_steps:
        engine.lbfgs(
            params,
            lambda: task.backward_full_batch(net),
            t.lbfgs_steps,
            t.lbfgs_eval_every,
            evaluate,
            log,
            save_log,
        )
        save_network()
    log["final"] = log["evals"][-1]
    save_log()
    if done:
        checkpoint.remove()
    return log


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m jaxley_pytorch.training",
        description="Train one arm of an experiment for one seed "
        "(see jaxley_pytorch/training/run.py and config.py).",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="the arm's TOML file"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="runs go to <output root>/<experiment_name>/seed<k>",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="one of the config's seeds (if it has several)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue the run from its checkpoint"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="at most 20 steps (a quick check of a config)",
    )
    parser.add_argument("--device", default=None, help="default: cuda if available")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = pick_seed(cfg, args.seed)
    if args.smoke:
        cfg = smoke(cfg)
    root = args.output_root.resolve()
    out = root / cfg.experiment_name / f"seed{seed}"
    if args.resume and finished(out):
        print(f"{out} is finished; nothing to resume", flush=True)
        return out
    resolved = resolve(cfg, seed, root, args.smoke)
    print(json.dumps(resolved), flush=True)
    prepare_run_dir(out, args.config, resolved, args.resume)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train(cfg, seed, out, root, device)
    print(f"run directory: {out}", flush=True)
    return out
