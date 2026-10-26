# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Run configurations: one TOML file per arm of an experiment, read by
`python -m jaxley_pytorch.training` (`run.py`).

    experiment_name = "n512_students"  # <output root>/<experiment_name>/seed<k>
    seeds = [1, 2, 3]                  # one run per seed; `--seed` picks it

    [model]    a network drawn from scratch: name = "blk_rate" (`rate_network.BLKRate`)
    [init]     instead of [model]: a trained run's network, the teacher (`InitConfig`)
    [task]     name = "cycling" or "text" (`tasks.Cycling`, `tasks.Text`)
    [trainer]  the training (`TrainerConfig`)

The keys of [model] and [task] are the arguments of the class that `name` selects. Every
setting that affects a run's results must be written in the file: only keys with a
default (in `InitConfig` and `TrainerConfig`) may be left out. Unknown keys are errors.
The run directory keeps the file (`config.toml`) and every value the run used
(`resolved_config.json`).
"""

import dataclasses
import inspect
import re
import tomllib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from jaxley_pytorch.training.init import STARTS
from jaxley_pytorch.training.rate_network import BLKRate
from jaxley_pytorch.training.tasks import TASKS

MODELS = {"blk_rate": BLKRate}
WEIGHTS = ("W", "g", "b")  # a network's own parameters (a task adds its io_names)
CLIP_MODES = ("global", "per_tensor")
DTYPES = ("float32", "float64")
_TOP_LEVEL = ("experiment_name", "seeds", "model", "init", "task", "trainer")


@dataclass(frozen=True)
class Component:
    """`name` selects an implementation; `kwargs` are the arguments it is built with."""

    name: str
    kwargs: Dict[str, Any]

    def settings(self) -> Dict:
        return {"name": self.name, **self.kwargs}


@dataclass(frozen=True)
class InitConfig:
    """`[init]`: the run's network is that of the trained run in the directory `teacher`
    (relative to the output root unless absolute), with gains and biases as `start` says
    (`init.STARTS`: "shuffled", "random", "uniform", "teacher" or "student"; for
    "student", `student` is a trained student's run directory, in which `{seed}` stands
    for the run's seed). `stable_start` (cycling; random or uniform starts): draw the
    start again while its activity runs away (`init.stable_student`)."""

    teacher: str
    start: str
    student: str = ""
    stable_start: bool = False

    def __post_init__(self):
        if self.start not in STARTS:
            raise ValueError(
                f"[init] start must be one of {STARTS}, got {self.start!r}"
            )
        if (self.start == "student") != bool(self.student):
            raise ValueError(
                '[init] student (a run directory) goes with start = "student", and '
                "only with it"
            )
        if self.stable_start and self.start not in ("random", "uniform"):
            raise ValueError('[init] stable_start is for start = "random" or "uniform"')


def _check(section: str, key: str, ok: bool, what: str, value):
    if not ok:
        raise ValueError(f"[{section}] {key} must be {what}, got {value!r}")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class TrainerConfig:
    """`[trainer]`: Adam on the parameters in `trainable` ("W", "g", "b" and the task's
    input and readout weights, its `io_names`), `steps` steps of `batch` trials or text
    streams. The learning rate `lr` warms up linearly over `warmup_steps` and then
    decays linearly to zero at the last step. The gradient norm is clipped at `clip`, of
    all the parameters together (`clip_mode = "global"`) or of every tensor on its own
    ("per_tensor"). An evaluation every `eval_every` steps (and before and after
    training), the training loss logged every `log_every`; `dtype` "float32" or
    "float64".

    Optional: a checkpoint every `checkpoint_every` steps (0: none; `--resume` continues
    from the last one); a time limit of `hours` (0: none); `lbfgs_steps` full-batch
    L-BFGS steps after Adam (cycling; an evaluation every `lbfgs_eval_every`);
    torch.compile (`compile`) and CUDA graphs (`cuda_graph`)."""

    trainable: Tuple[str, ...]
    steps: int
    warmup_steps: int
    batch: int
    lr: float
    clip: float
    clip_mode: str
    eval_every: int
    log_every: int
    dtype: str
    checkpoint_every: int = 0
    hours: float = 0.0
    lbfgs_steps: int = 0
    lbfgs_eval_every: int = 25
    compile: bool = True
    cuda_graph: bool = True

    def __post_init__(self):
        names = self.trainable
        _check(
            "trainer",
            "trainable",
            isinstance(names, (list, tuple)) and bool(names),
            'a list of parameter names, e.g. ["g", "b"]',
            names,
        )
        object.__setattr__(self, "trainable", tuple(names))
        for key in ("steps", "checkpoint_every", "lbfgs_steps"):
            v = getattr(self, key)
            _check("trainer", key, isinstance(v, int) and v >= 0, "an integer >= 0", v)
        for key in (
            "warmup_steps",
            "batch",
            "eval_every",
            "log_every",
            "lbfgs_eval_every",
        ):
            v = getattr(self, key)
            _check("trainer", key, isinstance(v, int) and v >= 1, "an integer >= 1", v)
        if self.steps and self.warmup_steps > self.steps:
            raise ValueError("[trainer] warmup_steps must be at most steps")
        for key in ("lr", "clip"):
            v = getattr(self, key)
            _check("trainer", key, _number(v) and v > 0, "a positive number", v)
        _check(
            "trainer",
            "hours",
            _number(self.hours) and self.hours >= 0,
            ">= 0",
            self.hours,
        )
        _check(
            "trainer",
            "clip_mode",
            self.clip_mode in CLIP_MODES,
            f"one of {CLIP_MODES}",
            self.clip_mode,
        )
        _check("trainer", "dtype", self.dtype in DTYPES, f"one of {DTYPES}", self.dtype)
        for key in ("compile", "cuda_graph"):
            v = getattr(self, key)
            _check("trainer", key, isinstance(v, bool), "true or false", v)


@dataclass(frozen=True)
class RunConfig:
    """One arm of an experiment (see the module's docstring)."""

    experiment_name: str
    seeds: Tuple[int, ...]
    task: Component
    trainer: TrainerConfig
    model: Optional[Component] = None
    init: Optional[InitConfig] = None

    def __post_init__(self):
        if (self.model is None) == (self.init is None):
            raise ValueError(
                "set either [model] (a network drawn from scratch) or [init] (the "
                "network of a trained run), not both"
            )
        valid = WEIGHTS + TASKS[self.task.name].io_names
        unknown = sorted(set(self.trainer.trainable) - set(valid))
        if unknown:
            raise ValueError(
                f"[trainer] trainable has unknown parameter(s) {unknown}; the "
                f"{self.task.name} task's networks have {list(valid)}"
            )
        if self.task.name != "cycling":
            if self.init is not None and self.init.stable_start:
                raise ValueError("[init] stable_start is for the cycling task")
            if self.trainer.lbfgs_steps:
                raise ValueError("[trainer] lbfgs_steps is for the cycling task")
        if (
            self.task.name == "text"
            and self.task.kwargs["eval_segments"] % self.trainer.batch
        ):
            raise ValueError(
                "[task] eval_segments must be a multiple of [trainer] batch"
            )


def _check_keys(section: str, body: Dict, allowed, required):
    unknown = sorted(set(body) - set(allowed))
    if unknown:
        raise ValueError(
            f"[{section}] has unknown key(s) {unknown}; valid keys are "
            f"{sorted(allowed)}"
        )
    missing = sorted(set(required) - set(body))
    if missing:
        raise ValueError(
            f"[{section}] is missing {missing} (every setting that affects the results "
            "must be in the file)"
        )


def _component(raw: Dict, section: str, registry: Dict) -> Component:
    body = dict(raw[section])
    name = body.pop("name", None)
    if name not in registry:
        raise ValueError(
            f"[{section}] name must be one of {sorted(registry)}, got {name!r}"
        )
    params = inspect.signature(registry[name]).parameters.values()
    required = [p.name for p in params if p.default is inspect.Parameter.empty]
    _check_keys(section, body, [p.name for p in params], required)
    return Component(name, body)


def _section(cls, raw: Dict, section: str):
    body = dict(raw[section])
    fields = dataclasses.fields(cls)
    required = [f.name for f in fields if f.default is dataclasses.MISSING]
    _check_keys(section, body, [f.name for f in fields], required)
    return cls(**body)


def load_config(path) -> RunConfig:
    """Read and check a run configuration; a ValueError names the offending section and
    key."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = sorted(set(raw) - set(_TOP_LEVEL))
    if unknown:
        raise ValueError(
            f"unknown top-level key(s) or section(s) {unknown}; valid ones are "
            f"{list(_TOP_LEVEL)}"
        )
    missing = [
        key for key in ("experiment_name", "seeds", "task", "trainer") if key not in raw
    ]
    if missing:
        raise ValueError(f"the config is missing {missing}")
    name = raw["experiment_name"]
    if not (
        isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
    ):
        raise ValueError(
            "experiment_name must be a directory name (letters, digits, '_', '-', "
            f"'.'), got {name!r}"
        )
    seeds = raw["seeds"]
    if not (
        isinstance(seeds, list)
        and seeds
        and all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds
        )
        and len(set(seeds)) == len(seeds)
    ):
        raise ValueError(
            f"seeds must be a list of distinct integers >= 0, got {seeds!r}"
        )
    return RunConfig(
        experiment_name=name,
        seeds=tuple(seeds),
        task=_component(raw, "task", TASKS),
        trainer=_section(TrainerConfig, raw, "trainer"),
        model=_component(raw, "model", MODELS) if "model" in raw else None,
        init=_section(InitConfig, raw, "init") if "init" in raw else None,
    )
