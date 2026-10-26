# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""The training loop every run shares (`fit`), its checkpoints (`Checkpoint`), and the
optional full-batch L-BFGS finish (`lbfgs`)."""

import json
import math
import os
import time
from typing import Callable, Dict, List

import numpy as np
import torch


def lr_factor(steps: int, warmup_steps: int) -> Callable[[int], float]:
    """The learning rate of step s (from 0) is `lr * factor(s)`: linear warmup over
    `warmup_steps`, then linear decay to zero at the last step."""

    def factor(s):
        if s < warmup_steps:
            return (s + 1) / warmup_steps
        return max(0.0, (steps - s) / (steps - warmup_steps))

    return factor


def make_optimizer(params: List[torch.Tensor], cfg, use_graph: bool):
    """Adam and its learning-rate schedule (`lr_factor`). With CUDA graphs the learning
    rate is a tensor, which the scheduler updates in place, so the change reaches the
    captured step."""
    lr = torch.tensor(cfg.lr, device=params[0].device) if use_graph else cfg.lr
    opt = torch.optim.Adam(params, lr=lr, capturable=use_graph)
    factor = lr_factor(cfg.steps, cfg.warmup_steps)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def clip_gradients(params: List[torch.Tensor], clip: float, mode: str):
    """Clip the gradient norm of all `params` together (`"global"`) or of every tensor
    on its own (`"per_tensor"`)."""
    if mode == "global":
        torch.nn.utils.clip_grad_norm_(params, clip, foreach=True)
    else:
        for p in params:
            torch.nn.utils.clip_grad_norm_([p], clip)


def on_side_stream(fn: Callable):
    """Run `fn()` on a side stream (everything that touches the parameters before the
    graph capture, as in examples/language_modeling/train.py)."""
    if not torch.cuda.is_available():
        return fn()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        result = fn()
    torch.cuda.current_stream().wait_stream(side)
    return result


class Checkpoint:
    """A run's checkpoint in the file `path`: the step, the optimizer, its learning-rate
    schedule and the log so far, and `get_state()` (the network's parameters and the
    task's position in its data), which `set_state` restores."""

    def __init__(self, path: str, get_state: Callable, set_state: Callable):
        self.path, self.get_state, self.set_state = path, get_state, set_state

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def save(self, step: int, opt, scheduler, log: Dict):
        state = {
            "step": step,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "log": log,
            "state": self.get_state(),
        }
        torch.save(state, self.path + ".tmp")
        os.replace(self.path + ".tmp", self.path)  # never a half-written checkpoint

    def restore(self, opt, scheduler, log: Dict) -> int:
        device = opt.param_groups[0]["params"][0].device
        ck = torch.load(self.path, map_location=device, weights_only=False)
        opt.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        log.clear()
        log.update(ck["log"])
        self.set_state(ck["state"])
        return ck["step"]

    def remove(self):
        if self.exists():
            os.remove(self.path)


def fit(
    cfg,
    params: List[torch.Tensor],
    loss_fn: Callable,
    carry: Callable,
    next_batch: Callable,
    evaluate: Callable,
    log: Dict,
    save_log: Callable,
    checkpoint: Checkpoint = None,
):
    """`cfg.steps` steps of Adam on `params` (`cfg` is a `config.TrainerConfig`).

    Every step, `next_batch()` fills the static inputs and `loss_fn()` returns the loss
    and the network state to carry to the next step (or None), which `carry(state)`
    stores after the update. The learning rate warms up linearly over
    `cfg.warmup_steps` and then decays linearly to zero; gradients are clipped at
    `cfg.clip`. On CUDA (`cfg.cuda_graph`), the step (simulation, loss, backward pass,
    clipping, Adam) is captured in a CUDA graph after two eager steps on a side stream
    and replayed, as in examples/language_modeling/train.py.

    `evaluate(step)` runs before training, every `cfg.eval_every` steps and at the end;
    `save_log()` then writes `log` (its "train", "evals" and "timing"). With
    `checkpoint`, the run is saved every `cfg.checkpoint_every` steps and when the time
    limit `cfg.hours` stops it, and resumes from the checkpoint if there is one.
    Returns False if the time limit stopped the run, True if it took every step."""
    n_steps = cfg.steps
    device = params[0].device
    use_graph = cfg.cuda_graph and device.type == "cuda"
    opt, scheduler = make_optimizer(params, cfg, use_graph)
    first = 0
    if checkpoint is not None and checkpoint.exists():
        first = checkpoint.restore(opt, scheduler, log)
        print(f"resumed at step {first}", flush=True)
    else:
        log["evals"].append(evaluate(0))
        print(json.dumps(log["evals"][-1]), flush=True)
        save_log()

    def iteration():
        opt.zero_grad(set_to_none=False)
        loss, state = loss_fn()
        loss.backward()
        clip_gradients(params, cfg.clip, cfg.clip_mode)
        opt.step()
        if state is not None:
            carry(state)
        return loss.detach()

    side = torch.cuda.Stream() if use_graph else None
    graph, static_loss = None, None
    losses = torch.zeros(n_steps, device=device)
    t_start, t_ready, step_times = time.time(), None, []
    for step in range(first, n_steps):
        lr = float(scheduler.get_last_lr()[0])
        next_batch()
        t0 = time.time()
        i = step - first  # steps of this process
        if graph is not None:
            graph.replay()
        elif use_graph and i < 2:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                static_loss = iteration()
            torch.cuda.current_stream().wait_stream(side)
        elif use_graph:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_loss = iteration()
            graph.replay()
        else:
            static_loss = iteration()
        losses[step] = static_loss
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        if t_ready is None and i == (2 if use_graph else 1):
            t_ready = time.time() - t_start
            print(
                f"ready after {t_ready:.1f} s (compilation, graph capture)", flush=True
            )
        last = step == n_steps - 1
        out_of_time = cfg.hours > 0 and time.time() - t_start > cfg.hours * 3600
        if (step + 1) % cfg.log_every == 0 or last or out_of_time:
            recent = (
                losses[max(first, step + 1 - cfg.log_every) : step + 1].mean().item()
            )
            s_step = (
                float(np.median(step_times[3:]))
                if len(step_times) > 4
                else step_times[-1]
            )
            print(
                f"step {step + 1}: loss {recent:.6g}, lr {lr:.3g}, {s_step:.4f} s/step",
                flush=True,
            )
            log["train"].append(
                {
                    "step": step + 1,
                    "loss": recent,
                    "lr": lr,
                    "elapsed_s": time.time() - t_start,
                }
            )
            if not math.isfinite(recent):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
        if (step + 1) % cfg.eval_every == 0 or last or out_of_time:
            log["evals"].append(evaluate(step + 1))
            print(json.dumps(log["evals"][-1]), flush=True)
            timed = step_times[3:] if len(step_times) > 4 else step_times
            cuda = device.type == "cuda"
            log["timing"] = {
                "steps": step + 1,
                "ready_s": t_ready,
                "s_per_step": float(np.median(timed)),
                "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9
                if cuda
                else None,
                "graph": use_graph,
            }
            save_log()
        every = cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0
        if checkpoint is not None and not last and (every or out_of_time):
            checkpoint.save(step + 1, opt, scheduler, log)
        if out_of_time:
            print(f"time limit reached after {step + 1} steps", flush=True)
            return False
    return True


def lbfgs(
    params: List[torch.Tensor],
    backward_full_batch: Callable,
    steps: int,
    eval_every: int,
    evaluate: Callable,
    log: Dict,
    save_log: Callable,
):
    """`steps` steps of full-batch L-BFGS (strong-Wolfe line search) on `params`:
    `backward_full_batch()` accumulates the gradients of the full-batch loss and returns
    the loss. No stopping tolerances (always `steps` steps; the gradient norm is
    logged). The evaluations, every `eval_every` steps and at the end, are marked
    `"phase": "lbfgs"`."""
    opt = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=1,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )

    def closure():
        opt.zero_grad()
        return backward_full_batch()

    first = log["evals"][-1]["step"]
    for it in range(steps):
        loss = float(opt.step(closure))
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at L-BFGS step {it + 1}")
        if (it + 1) % eval_every == 0 or it == steps - 1:
            grad_norm = float(torch.cat([p.grad.flatten() for p in params]).norm())
            ev = dict(evaluate(first + it + 1), phase="lbfgs", grad_norm=grad_norm)
            log["evals"].append(ev)
            print(json.dumps(ev), flush=True)
            save_log()
