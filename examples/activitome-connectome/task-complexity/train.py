"""Teacher-student experiment: does a harder task pin down single-neuron activity? (see
README.md). One script, three roles:

  check    the jaxley_pytorch network against the paper's update equation: rates and the
           gradients of a random function of them (float64, random parameters).
  teacher  train a network on `--task` (cycling | chars | tokens): recurrent weights,
           gains, biases and the task's input and readout weights (the cycling task's
           input weights stay fixed, as in the paper). Writes `teacher.pt` and
           `result.json` to `--out`.
  student  load the teacher in `--teacher`, start from its gains and biases permuted
           within the excitatory and within the inhibitory neurons, gains halved (the
           paper's `nn_fig1_trainStudent.py`), and train only these on the task itself:
           the cycling task's target (mean squared error) or the next token of the text
           (cross-entropy); the teacher's outputs are not used. Everything else is the
           teacher's. `--init teacher` starts from the teacher's own gains and biases
           instead (a control: how far training alone moves a student away from the
           teacher). Writes `student.pt` and `result.json` (the errors against the
           teacher during training).

Every run (teacher or student) warms its learning rate up linearly over the first 5% of
the steps and then decays it linearly to zero, so that it ends at an optimum of its loss.
On the GPU every training step (simulation, loss, backward pass, Adam) is captured in a
CUDA graph after two eager steps and replayed, as in examples/language_modeling/train.py.

Usage:
  python train.py --role check --out <dir>
  python train.py --role teacher --task chars --out <runs>/chars/teacher
  python train.py --role student --task chars --teacher <runs>/chars/teacher --seed 1 --out <runs>/chars/student_seed1
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rate_network as rn
import tasks

LN2 = math.log(2.0)
# Cycling: the paper's recipes (teacher 3,000 steps instead of its 1,400). Text teachers:
# the recipe of examples/language_modeling. Text students: the paper's student recipe.
# Text steps are 32 x 256 tokens: teachers see 205M tokens, students 82M.
RECIPES = {
    ("cycling", "teacher"): dict(steps=3000, lr=1e-3, clip=5.0, clip_mode="global", eval_every=100, log_every=50),
    ("cycling", "student"): dict(steps=7000, lr=1e-3, clip=2.0, clip_mode="global", eval_every=50, log_every=50),
    ("text", "teacher"): dict(steps=25_000, lr=3e-3, clip=1.0, clip_mode="per-param", eval_every=1000, log_every=100),
    ("text", "student"): dict(steps=10_000, lr=1e-3, clip=2.0, clip_mode="global", eval_every=250, log_every=50),
}
WARMUP = 0.05  # every run: linear warmup over this fraction of the steps, then linear decay to zero


def kind(task: str) -> str:
    return "cycling" if task == "cycling" else "text"


def lr_factor(recipe):
    """The learning rate of step s (from 0) is `lr * factor(s)`: linear warmup, then
    linear decay to zero at the last step."""
    n = recipe["steps"]
    w = max(1, int(round(recipe["warmup"] * n)))

    def factor(s):
        return (s + 1) / w if s < w else max(0.0, (n - s) / (n - w))

    return factor


def make_optimizer(params, recipe, use_graph):
    """Adam and its learning-rate schedule (`lr_factor`). With CUDA graphs the learning
    rate is a tensor, which the scheduler updates in place, so the change reaches the
    captured step."""
    lr = torch.tensor(recipe["lr"], device=params[0].device) if use_graph else recipe["lr"]
    opt = torch.optim.Adam(params, lr=lr, capturable=use_graph)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(recipe))


def clip(params, recipe):
    if recipe["clip_mode"] == "global":
        torch.nn.utils.clip_grad_norm_(params, recipe["clip"], foreach=True)
    else:  # Every tensor on its own.
        for p in params:
            torch.nn.utils.clip_grad_norm_([p], recipe["clip"])


# ----------------------------------------------------------------------------------
# Networks


def new_teacher(args, device, dtype, vocab=None):
    rng = np.random.default_rng(args.seed)
    g, b, mask, w, n_exc = rn.paper_init(args.n, rng)
    conn, w_edges = rn.edges_from_mask(mask, w, n_exc)
    n, k = args.n, args.width
    if args.task == "cycling":
        io = {"w_in": rng.standard_normal((2, n)) / rn.ALPHA, "w_out": 5 * rng.standard_normal(n) / n}
        trainable = {"W", "g", "b", "w_out"}
    else:
        io = {"embed": rng.standard_normal((vocab, k)), "embed_proj": rng.standard_normal((n, k)) / math.sqrt(k),
              "read_proj": rng.standard_normal((k, n)) / math.sqrt(n),
              "unembed": rng.standard_normal((vocab, k)) / math.sqrt(k), "read_bias": np.zeros(vocab)}
        trainable = {"W", "g", "b"} | set(io)
    t0 = time.time()
    sim = rn.Simulator(conn, device, dtype, compile=bool(args.compile))
    print(f"network: {n} neurons ({n_exc} excitatory), {int(conn['real'].sum()):,} synapses "
          f"(padded to {conn['k']} per neuron), built in {time.time() - t0:.1f} s", flush=True)
    return rn.RateRNN(sim, w_edges, g, b, io, trainable), conn


def save_teacher(path, model, conn, args):
    torch.save({"conn": {k: torch.as_tensor(v) for k, v in conn.items()},
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "task": args.task, "args": vars(args)}, path)


def load_teacher(run_dir, device, dtype, compile):
    ck = torch.load(os.path.join(run_dir, "teacher.pt"), map_location="cpu", weights_only=True)
    conn = {k: (v.item() if v.dim() == 0 else v.numpy()) for k, v in ck["conn"].items()}
    sim = rn.Simulator(conn, device, dtype, compile=compile)
    st = ck["state"]
    io = {k[len("io."):]: v for k, v in st.items() if k.startswith("io.")}
    return rn.RateRNN(sim, st["W"], st["g"], st["b"], io, trainable=()), conn, ck


def make_student(teacher, conn, seed, init="shuffled"):
    """A student with the teacher's weights, inputs and readout. Gains and biases:
    `shuffled`, the paper's student initialisation (the teacher's, each permuted within
    the excitatory and within the inhibitory neurons, gains halved); `teacher`, the
    teacher's own."""
    g = teacher.g.detach().cpu().double().numpy().copy()
    b = teacher.b.detach().cpu().double().numpy().copy()
    if init == "shuffled":
        rng = np.random.default_rng(seed)
        for part in (slice(0, conn["n_exc"]), slice(conn["n_exc"], None)):
            g[part] = rng.permutation(g[part])
            b[part] = rng.permutation(b[part])
        g = 0.5 * g
    io = {k: v.detach() for k, v in teacher.io.items()}
    return rn.RateRNN(teacher.sim, teacher.W.detach(), g, b, io, trainable={"g", "b"})


# ----------------------------------------------------------------------------------
# Evaluation


@torch.no_grad()
def eval_cycling(model, teacher, inputs, batch):
    """Readout (trials, steps) and rates (trials, steps, n) of `model` (and `teacher`) on
    the trials `inputs`, in batches of `batch` (the last one padded)."""
    n_trials = inputs.shape[0]
    pad = (-n_trials) % batch
    inp = torch.cat([inputs, inputs[:pad]]) if pad else inputs
    out = {}
    for name, net in (("s", model), ("t", teacher)):
        if net is None:
            continue
        zs, rates = [], []
        for i in range(0, len(inp), batch):
            z, a, _ = net.run_trials(inp[i : i + batch], net.sim.rest_states(batch))
            zs.append(z)
            rates.append(a)
        out[f"z_{name}"] = torch.cat(zs)[:n_trials]
        out[f"a_{name}"] = torch.cat(rates)[:n_trials].transpose(1, 2)
    return out


@torch.no_grad()
def eval_text(model, teacher, tokens, burn, batch):
    """Window-mean rates (segments, tokens after `burn`, n) of `model` (and `teacher`) on
    `tokens` (segments, length + 1), every segment from rest, and the summed
    cross-entropies (and KL divergence from the teacher) of the tokens after `burn` (nats)."""
    out = {"ce_s": 0.0, "ce_t": 0.0, "kl": 0.0, "count": 0}
    rates = {"s": [], "t": []}
    for i in range(0, tokens.shape[0], batch):
        tok = tokens[i : i + batch]
        target = tok[:, 1 + burn :].reshape(-1)
        logp = {}
        for name, net in (("s", model), ("t", teacher)):
            if net is None:
                continue
            logits, feats, _ = net.run_text(tok[:, :-1], net.sim.rest_states(tok.shape[0]))
            logp[name] = F.log_softmax(logits[:, burn:].float(), dim=-1)
            out[f"ce_{name}"] += F.nll_loss(logp[name].reshape(-1, logits.shape[-1]), target, reduction="sum").item()
            rates[name].append(feats[:, burn:])
        if teacher is not None:
            out["kl"] += (logp["t"].exp() * (logp["t"] - logp["s"])).sum().item()
        out["count"] += target.numel()
    for name, r in rates.items():
        if r:
            out[f"a_{name}"] = torch.cat(r)
    return out


def _corr_cols(x, y):
    """Pearson correlation of every column of x with the same column of y (nan if one is
    constant)."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    return (x * y).sum(0) / torch.sqrt((x * x).sum(0) * (y * y).sum(0))


@torch.no_grad()
def activity_stats(a, basis):
    """Dimensionality of rates `a` (..., n): participation ratio, components for 90% and
    95% of the variance, and the fraction of the variance in the readout's directions."""
    x = a.reshape(-1, a.shape[-1]).double()
    x = x - x.mean(0)
    cov = x.T @ x / (x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    frac = torch.cumsum(ev, 0) / ev.sum()
    return {"participation_ratio": float(ev.sum() ** 2 / (ev**2).sum()),
            "components_90": int((frac < 0.9).sum()) + 1, "components_95": int((frac < 0.95).sum()) + 1,
            "readout_rank": int(basis.shape[1]),
            "readout_variance_fraction": float(torch.trace(basis.T @ cov @ basis) / ev.sum()),
            "mean_rate": float(a.mean())}


@torch.no_grad()
def baselines(a_t, teacher, basis, n_perm=100, seed=0):
    """The paper's shuffled baselines: the errors of the teacher's own rates and parameters
    against a random permutation of its neurons (mean over `n_perm` permutations)."""
    gen = torch.Generator().manual_seed(seed)
    n = a_t.shape[-1]
    x = a_t.reshape(-1, n).double()
    g, b = torch.relu(teacher.g).detach().double(), teacher.b.detach().double()
    acc = {k: 0.0 for k in ["act_mse", "act_mse_readout", "act_corr_err", "gain_rmse", "bias_rmse"]}
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=gen).to(x.device)
        e = x[:, perm] - x
        acc["act_mse"] += float((e**2).mean())
        acc["act_mse_readout"] += float(((e @ basis) ** 2).sum(-1).mean() / n)
        acc["act_corr_err"] += float(1 - torch.nanmean(_corr_cols(x[:, perm], x).abs()))
        acc["gain_rmse"] += float(torch.sqrt(((g[perm] - g) ** 2).mean()))
        acc["bias_rmse"] += float(torch.sqrt(((b[perm] - b) ** 2).mean()))
    out = {f"{k}_shuffled": v / n_perm for k, v in acc.items()}
    out["act_mse_other_shuffled"] = out["act_mse_shuffled"] - out["act_mse_readout_shuffled"]
    return out


@torch.no_grad()
def compare(a_s, a_t, student, teacher, basis):
    """Errors of the student against the teacher (the paper's Fig. 1d,e): rates (mean
    squared error, split into the readout's directions and the rest; 1 - mean |Pearson
    r|), gains (RMS of relu(g)) and biases (RMS)."""
    n = a_t.shape[-1]
    xs, xt = a_s.reshape(-1, n).double(), a_t.reshape(-1, n).double()
    e = xs - xt
    out = {"act_mse": float((e**2).mean()), "act_mse_readout": float(((e @ basis) ** 2).sum(-1).mean() / n),
           "act_corr_err": float(1 - torch.nanmean(_corr_cols(xs, xt).abs())),
           "gain_rmse": float(torch.sqrt(((torch.relu(student.g) - torch.relu(teacher.g)).double() ** 2).mean())),
           "bias_rmse": float(torch.sqrt(((student.b - teacher.b).double() ** 2).mean()))}
    out["act_mse_other"] = out["act_mse"] - out["act_mse_readout"]
    return out


# ----------------------------------------------------------------------------------
# Training loop


def fit(args, recipe, scheduler, params, iteration, load_batch, evaluate, log, out_path):
    """`recipe["steps"]` training steps: `load_batch()` fills the static inputs,
    `iteration()` does one step and returns its loss (a tensor); `scheduler` sets the
    learning rate after every step. On CUDA the step is captured in a CUDA graph after
    two eager steps (on a side stream). `evaluate(step)` runs before training, every
    `eval_every` steps and at the end."""
    n_steps = recipe["steps"]
    device = params[0].device
    use_graph = bool(args.graph) and device.type == "cuda"
    side = torch.cuda.Stream() if use_graph else None
    graph, static_loss = None, None
    losses = torch.zeros(n_steps, device=device)
    t_start, t_ready, step_times = time.time(), None, []
    log["evals"].append(evaluate(0))
    print(json.dumps(log["evals"][-1]), flush=True)
    for step in range(n_steps):
        lr = float(scheduler.get_last_lr()[0])
        load_batch()
        t0 = time.time()
        if graph is not None:
            graph.replay()
        elif use_graph and step < 2:
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
        if t_ready is None and step == (2 if use_graph else 1):
            t_ready = time.time() - t_start
            print(f"ready after {t_ready:.1f} s (compilation and graph capture)", flush=True)
        last = step == n_steps - 1
        out_of_time = args.hours is not None and time.time() - t_start > args.hours * 3600
        if (step + 1) % recipe["log_every"] == 0 or last or out_of_time:
            recent = losses[max(0, step + 1 - recipe["log_every"]) : step + 1].mean().item()
            s_step = float(np.median(step_times[3:])) if len(step_times) > 4 else step_times[-1]
            print(f"step {step + 1}: loss {recent:.6g}, lr {lr:.3g}, {s_step:.4f} s/step", flush=True)
            log["train"].append({"step": step + 1, "loss": recent, "lr": lr, "elapsed_s": time.time() - t_start})
            if not math.isfinite(recent):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
        if (step + 1) % recipe["eval_every"] == 0 or last or out_of_time:
            log["evals"].append(evaluate(step + 1))
            print(json.dumps(log["evals"][-1]), flush=True)
            timed = step_times[3:] if len(step_times) > 4 else step_times
            log["timing"] = {"steps": step + 1, "ready_s": t_ready, "s_per_step": float(np.median(timed)),
                             "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9
                             if device.type == "cuda" else None, "graph": use_graph}
            with open(out_path, "w") as f:  # partial, overwritten at every evaluation
                json.dump(log, f, indent=1)
        if out_of_time:
            print(f"time limit reached after {step + 1} steps", flush=True)
            break


def on_side_stream(fn):
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


# ----------------------------------------------------------------------------------
# Roles


def run_check(args, device):
    """Rates and gradients of the jaxley_pytorch network vs the paper's update."""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    rng = np.random.default_rng(args.seed)
    g, b, mask, w, n_exc = rn.paper_init(args.n, rng)
    conn, w_edges = rn.edges_from_mask(mask, w, n_exc)
    sim = rn.Simulator(conn, device, dtype, compile=bool(args.compile))
    model = rn.RateRNN(sim, w_edges, g, b, {}, trainable={"W", "g", "b"})
    batch, steps = 4, 200
    gen = torch.Generator().manual_seed(args.seed)
    current = (2.0 * torch.randn(batch, args.n, steps, generator=gen)).to(device)
    probe = torch.randn(batch, args.n, steps, generator=gen).to(device)
    params = [model.W, model.g, model.b]

    v, _ = sim.segment(sim.rest_states(batch), current, sim.edge_params(*params))
    a_jx = model.rates(v)
    grads_jx = torch.autograd.grad((a_jx * probe).sum(), params)

    J = sim.dense_weights(model.W)
    x = torch.zeros(batch, args.n, device=device)
    gain = torch.relu(model.g)
    rates = []
    for t in range(steps):
        x = x + rn.ALPHA * (-x + (gain * F.softplus(x + model.b)) @ J.T + current[..., t])
        rates.append(gain * F.softplus(x + model.b))
    a_ref = torch.stack(rates, dim=-1)
    grads_ref = torch.autograd.grad((a_ref * probe).sum(), params)

    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    out = {"rates_rel_err": rel(a_jx, a_ref), "mean_rate": float(a_ref.mean()), "max_rate": float(a_ref.max()),
           **{f"grad_{name}_rel_err": rel(gj, gr) for name, gj, gr in zip(["W", "g", "b"], grads_jx, grads_ref)}}
    out["passed"] = all(v < 1e-9 for k, v in out.items() if k.endswith("rel_err"))
    print(json.dumps(out), flush=True)
    return out


def run_teacher(args, device, dtype):
    recipe = recipe_of(args)
    use_graph = bool(args.graph) and device.type == "cuda"
    B = args.batch
    if args.task == "cycling":
        model, conn = new_teacher(args, device, dtype)
        inputs, targets = (torch.as_tensor(x, device=device, dtype=dtype) for x in tasks.cycling_trials())
        static_in = torch.zeros(B, tasks.TRIAL_STEPS, 2, device=device)
        static_tg = torch.zeros(B, tasks.TRIAL_STEPS, device=device)
        rng = np.random.default_rng(args.seed + 1)

        def load_batch():
            idx = torch.as_tensor(rng.choice(tasks.TRIALS, B, replace=False), device=device)
            static_in.copy_(inputs[idx])
            static_tg.copy_(targets[idx])

        def forward_loss():
            z, _, _ = model.run_trials(static_in, model.sim.rest_states(B))
            return ((z - static_tg) ** 2).mean()

        def evaluate(step):  # task loss: mean squared error
            ev = eval_cycling(model, None, inputs, B)
            return {"step": step, "task_loss": float(((ev["z_s"] - targets) ** 2).mean()),
                    **activity_stats(ev["a_s"], model.readout_basis())}
    else:
        data = args.data or tasks.DATA[args.task]
        meta = tasks.load_meta(data)
        vocab, cpt = meta["vocab_size"], tasks.chars_per_token(meta)
        model, conn = new_teacher(args, device, dtype, vocab=vocab)
        streams = tasks.Streams(os.path.join(data, "train.bin"), meta["dtype"], B, args.chunk, seed=args.seed)
        eval_tok = torch.as_tensor(tasks.eval_segments(os.path.join(data, "val.bin"), meta["dtype"], args.eval_segments,
                                                       args.eval_burn + args.eval_length), device=device)
        static_tok = torch.zeros(B, args.chunk + 1, dtype=torch.long, device=device)
        static_states = on_side_stream(lambda: model.sim.rest_states(B))

        def load_batch():
            static_tok.copy_(torch.as_tensor(streams.next()))

        def forward_loss():
            logits, _, new = model.run_text(static_tok[:, :-1], static_states)
            loss = F.cross_entropy(logits.reshape(-1, vocab), static_tok[:, 1:].reshape(-1))
            return loss, new

        def evaluate(step):  # task loss: validation cross-entropy, bits per token
            ev = eval_text(model, None, eval_tok, args.eval_burn, B)
            bits = ev["ce_s"] / ev["count"] / LN2
            return {"step": step, "task_loss": bits, "val_bits_per_char": bits / cpt,
                    **activity_stats(ev["a_s"], model.readout_basis())}

    params = [p for p in model.parameters() if p.requires_grad]
    opt, scheduler = make_optimizer(params, recipe, use_graph)

    def iteration():
        opt.zero_grad(set_to_none=False)
        if args.task == "cycling":
            loss = forward_loss()
        else:
            loss, new = forward_loss()
        loss.backward()
        clip(params, recipe)
        opt.step()
        if args.task != "cycling":
            with torch.no_grad():
                for k in static_states:
                    static_states[k].copy_(new[k])
        return loss.detach()

    log = run_info(args, recipe, model, device)
    fit(args, recipe, scheduler, params, iteration, load_batch, evaluate, log, os.path.join(args.out, "result.json"))
    save_teacher(os.path.join(args.out, "teacher.pt"), model, conn, args)
    log["final"] = log["evals"][-1]
    return log


def run_student(args, device, dtype):
    recipe = recipe_of(args)
    use_graph = bool(args.graph) and device.type == "cuda"
    B = args.batch
    teacher, conn, ck = load_teacher(args.teacher, device, dtype, bool(args.compile))
    assert ck["task"] == args.task, (ck["task"], args.task)
    student = make_student(teacher, conn, args.seed, args.init)
    basis = teacher.readout_basis()
    if args.task == "cycling":
        inputs, targets = (torch.as_tensor(x, device=device, dtype=dtype) for x in tasks.cycling_trials())
        ev_t = eval_cycling(teacher, None, inputs, B)
        z_t, a_t = ev_t["z_s"], ev_t["a_s"]
        teacher_loss = float(((z_t - targets) ** 2).mean())
        static_in = torch.zeros(B, tasks.TRIAL_STEPS, 2, device=device)
        static_tg = torch.zeros(B, tasks.TRIAL_STEPS, device=device)
        rng = np.random.default_rng(args.seed + 1)

        def load_batch():
            idx = torch.as_tensor(rng.choice(tasks.TRIALS, B, replace=False), device=device)
            static_in.copy_(inputs[idx])
            static_tg.copy_(targets[idx])

        def forward_loss():
            z, _, _ = student.run_trials(static_in, student.sim.rest_states(B))
            return ((z - static_tg) ** 2).mean(), None

        def evaluate(step):  # task loss: mean squared error; output vs teacher: readout MSE
            ev = eval_cycling(student, None, inputs, B)
            return {"step": step, "task_loss": float(((ev["z_s"] - targets) ** 2).mean()),
                    "teacher_task_loss": teacher_loss, "output_vs_teacher": float(((ev["z_s"] - z_t) ** 2).mean()),
                    **compare(ev["a_s"], a_t, student, teacher, basis)}
    else:
        data = args.data or tasks.DATA[args.task]
        meta = tasks.load_meta(data)
        vocab, cpt = meta["vocab_size"], tasks.chars_per_token(meta)
        streams = tasks.Streams(os.path.join(data, "train.bin"), meta["dtype"], B, args.chunk, seed=args.seed)
        eval_tok = torch.as_tensor(tasks.eval_segments(os.path.join(data, "val.bin"), meta["dtype"], args.eval_segments,
                                                       args.eval_burn + args.eval_length), device=device)
        a_t = eval_text(teacher, None, eval_tok, args.eval_burn, B)["a_s"]
        static_tok = torch.zeros(B, args.chunk + 1, dtype=torch.long, device=device)
        static_states = on_side_stream(lambda: student.sim.rest_states(B))

        def load_batch():
            static_tok.copy_(torch.as_tensor(streams.next()))

        def forward_loss():
            logits, _, new = student.run_text(static_tok[:, :-1], static_states)
            return F.cross_entropy(logits.reshape(-1, vocab), static_tok[:, 1:].reshape(-1)), new

        def evaluate(step):  # task loss: validation cross-entropy; output vs teacher: KL (both bits per token)
            ev = eval_text(student, teacher, eval_tok, args.eval_burn, B)
            n = ev["count"]
            return {"step": step, "task_loss": ev["ce_s"] / n / LN2, "teacher_task_loss": ev["ce_t"] / n / LN2,
                    "val_bits_per_char": ev["ce_s"] / n / LN2 / cpt, "output_vs_teacher": ev["kl"] / n / LN2,
                    **compare(ev["a_s"], a_t, student, teacher, basis)}

    params = [student.g, student.b]
    opt, scheduler = make_optimizer(params, recipe, use_graph)

    def iteration():
        opt.zero_grad(set_to_none=False)
        loss, new = forward_loss()
        loss.backward()
        clip(params, recipe)
        opt.step()
        if new is not None:  # Text: carry the streams' state to the next step.
            with torch.no_grad():
                for k in static_states:
                    static_states[k].copy_(new[k])
        return loss.detach()

    log = run_info(args, recipe, student, device)
    log["teacher"] = {"dir": args.teacher, **activity_stats(a_t, basis)}
    log["baselines"] = baselines(a_t, teacher, basis)
    print(json.dumps({"teacher": log["teacher"], "baselines": log["baselines"]}), flush=True)
    fit(args, recipe, scheduler, params, iteration, load_batch, evaluate, log, os.path.join(args.out, "result.json"))
    torch.save({"g": student.g.detach().cpu(), "b": student.b.detach().cpu(), "seed": args.seed, "init": args.init},
               os.path.join(args.out, "student.pt"))
    log["final"] = log["evals"][-1]
    return log


def recipe_of(args):
    recipe = dict(RECIPES[(kind(args.task), args.role)], warmup=WARMUP)
    for key in ["steps", "lr", "eval_every", "log_every"]:
        if getattr(args, key) is not None:
            recipe[key] = getattr(args, key)
    return recipe


def run_info(args, recipe, model, device):
    return {"args": vars(args), "recipe": recipe, "task": args.task, "role": args.role,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "synapses": int(model.sim.real.sum()), "k": model.sim.k,
            "trainable": {name: p.numel() for name, p in model.named_parameters() if p.requires_grad},
            "train": [], "evals": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["check", "teacher", "student"], required=True)
    parser.add_argument("--task", choices=["cycling", "chars", "tokens"], default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher", default=None, help="the teacher's run directory (--role student)")
    parser.add_argument("--seed", type=int, default=0, help="teacher: network; student: initialisation and text")
    parser.add_argument("--init", choices=["shuffled", "teacher"], default="shuffled",
                        help="student: initial gains and biases (see make_student)")
    parser.add_argument("--data", default=None, help="token files (default per task, tasks.DATA)")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--width", type=int, default=75, help="text: width of the input and readout maps")
    parser.add_argument("--batch", type=int, default=32, help="trials or text streams per step")
    parser.add_argument("--chunk", type=int, default=256, help="text: tokens per truncated-BPTT step")
    parser.add_argument("--eval-segments", type=int, default=64, help="text: validation segments")
    parser.add_argument("--eval-burn", type=int, default=32, help="text: tokens before the measured ones")
    parser.add_argument("--eval-length", type=int, default=256, help="text: measured tokens per segment")
    parser.add_argument("--steps", type=int, default=None, help="override the recipe")
    parser.add_argument("--lr", type=float, default=None, help="override the recipe")
    parser.add_argument("--eval-every", type=int, default=None, help="override the recipe")
    parser.add_argument("--log-every", type=int, default=None, help="override the recipe")
    parser.add_argument("--hours", type=float, default=None, help="stop after this long")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.role != "check" and args.task is None:
        parser.error("--task is required")
    if args.role == "student" and args.teacher is None:
        parser.error("--teacher is required")
    if args.task in ("chars", "tokens") and args.eval_segments % args.batch:
        parser.error("--eval-segments must be a multiple of --batch")
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    if args.role == "check":
        result = run_check(args, device)
    else:
        dtype = getattr(torch, args.dtype)
        torch.set_default_dtype(dtype)
        result = run_teacher(args, device, dtype) if args.role == "teacher" else run_student(args, device, dtype)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
