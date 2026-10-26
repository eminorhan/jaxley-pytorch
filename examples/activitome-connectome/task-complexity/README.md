# Does a harder task pin down single-neuron activity?

Beiran & Litwin-Kumar (2025, *Nat. Neurosci.* 28:2561, code at
[emebeiran/connconstr](https://github.com/emebeiran/connconstr)) train "student"
networks that share a "teacher's" synaptic weights but not its single-neuron gains and
biases. Trained on a simple task (the cycling task, their Fig. 1), the students solve
the task with single-neuron activity no closer to the teacher's than the teacher's own
activity with shuffled neuron identities: the connectome plus the task leave the
solution degenerate. This experiment asks whether that holds for a much richer task:
next-token prediction on TinyStories, at the level of characters and of BPE tokens.

## Design

The paper's Fig. 1 protocol, with the student trained on the task itself:

* **Network** (`jaxley_pytorch.training.rate_network`, `[model] name = "blk_rate"`): the
  paper's rate network in jaxley_pytorch. 300 neurons (70% excitatory), each a passive
  compartment (time constant tau = 10 ms) whose rate is `relu(g) * softplus(x + b)`,
  current-based softplus synapses, every ordered pair connected with probability 0.5,
  Dale's law, the paper's initialisation and stability test. Forward Euler with
  dt = 0.5 ms is exactly the update of the paper's code (alpha = 0.05);
  `tests/training/test_rate_network.py` checks this against a direct implementation of
  that update (rates and gradients agree to 1e-15 in float64).
* **Tasks** (`jaxley_pytorch.training.tasks`):
  * `cycling`: the paper's task (60 fixed trials of 20 tau; a pulse on one of two inputs
    starts an oscillation of 2 or 5 periods in the one-dimensional readout).
  * `chars`: TinyStories at the level of characters, the files of
    `examples/language_modeling` (74 ASCII characters and an end-of-story token: every
    byte that occurs in the dataset, so this is the byte-level version).
  * `tokens`: the same text in byte-level BPE tokens (4,096, trained on the training split
    by `prepare_tokens.py`; 3.98 characters per token).

  Every character or token is a constant input current for 10 solver steps (tau / 2).
  Both text tasks read in and out through 75 dimensions: the input current of a token
  is `embed_proj @ embed[token]` and the logits are `unembed @ (read_proj @ a) + bias`
  (`a`: the rates averaged over the token's window), with 75-wide `embed` and
  `read_proj`. For characters this is no restriction (75 symbols); it makes the two text
  tasks differ only in their vocabulary.
* **Teachers** (`configs/<task>_teacher.toml`): recurrent weights, gains, biases and the
  readout trained together (and, for text, the input map); the cycling task's input
  weights stay random, as in the paper. Cycling: the paper's recipe (Adam 1e-3, batches
  of 32 of the 60 trials), 3,000 steps instead of 1,400. Text: 32 streams through the
  training text, backpropagation through windows of 256 tokens (the state carries over
  from window to window), Adam 3e-3 with every tensor's gradient clipped at 1 (as in
  `examples/language_modeling`), 25,000 steps (205M tokens). An average story is 803
  characters or 202 BPE tokens.
* **Students** (`configs/<task>_students.toml`, ten per task): the teacher's weights, inputs,
  readout and initial state; gains and biases initialised as in the paper's code (the
  teacher's, permuted within the excitatory and within the inhibitory neurons, gains
  halved), and only these trained (Adam 1e-3, gradient norm clipped at 2) on the task
  itself: the cycling task's target (mean squared error, 15,000 steps: after the paper's
  7,000 they were still learning the task) or the next token of the text (cross-entropy,
  10,000 steps of 32 x 256 tokens). The teacher's outputs are not used. One **control
  student** per task (`configs/<task>_control.toml`) starts from the teacher's own gains
  and biases: how far training alone moves a student away from the teacher sets the floor
  of the activity error. For the cycling task, ten more students start from new random
  gains and biases, drawn like the teacher's initial ones
  (`configs/cycling_students_random.toml`).
* **Learning rate**, every run: linear warmup over the first 5% of the steps, then linear
  decay to zero (`torch.optim.lr_scheduler.LambdaLR`), so that teachers and students end
  at an optimum of their loss.
* **Measurements**, during training, against the teacher (the paper's Fig. 1c-e): task
  loss next to the teacher's; how closely the output matches the teacher's (readout MSE,
  or KL divergence of the next-token distributions); the mean squared error of the rates
  (`relu(g) * softplus(x + b)`) of the same neurons, on the 60 trials or on 64 validation
  segments of 256 tokens (after 32 tokens from rest); 1 - mean |Pearson r| per neuron;
  the RMS errors of gains and biases; each next to its shuffled baseline (the teacher
  against a permutation of its own neurons). The rate error is also split into the
  readout's directions (1 for cycling, 75 for text) and the other directions, and the
  teacher's activity dimensionality (participation ratio, components for 90% of the
  variance) is recorded.

Not as in the paper: the students learn the task, not the teacher's readout (for the
cycling task, which the teacher solves almost perfectly, this is nearly the same); the
learning rate is scheduled (the paper's is constant); no noise (the paper adds 0.002 per
step during training); every trial and text stream starts from rest (x = 0) instead of
from a trained initial state; the synapses of a neuron are padded with zero-weight
synapses to the largest in-degree (the layout jaxley_pytorch steps fastest), which
changes no dynamics.

## Reading the results

R = (rate error after training) / (shuffled baseline), median over the ten students. A
student solves the task if its task loss is within 2% of the teacher's (text) or at most
twice the teacher's (cycling).

* `cycling` should reproduce the paper: R >= 1. If it does not, the comparison is moot.
* A text task supports the hypothesis (a richer task constrains the solution) if its
  students solve it with R <= 0.1, and contradicts it if R >= 0.5.
* The control student's R is the floor: students at that level match the teacher as well
  as the teacher itself does after the same training.
* Before training, students are already below the baseline (R ~ 0.3 in the smoke test:
  they start from the teacher's own values, shuffled, with the same wiring and inputs),
  so the change from before to after training is what counts; `summarize.py` reports
  both.

## Running

Every arm is one config file in `configs/` with all its settings, run by
`python -m jaxley_pytorch.training --config configs/<arm>.toml --output-root <dir> [--seed k]`
(`jaxley_pytorch/training/`; `--smoke` makes a 20-step check of a config). Outputs go to
`/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity`
(`data/`, `runs/<experiment_name>/seed<k>/`, `jobs/`, `figures/`, `probes/`). Jobs run on a
snapshot of the repo (`probes/jaxley_pytorch/tools/snapshot.sh`) with `~/myvenv`.

```bash
bash submit.sh         # 3 teachers; then per task 10 students and a control, and the cycling task's random-start students
bash submit_stress.sh  # the stress test (after submit.sh)
```

The BPE token files come from `prepare_tokens.py` (one CPU job, 2 min:
`python prepare_tokens.py --parquet <tinystories_gpt4_clean.parquet> --vocab 4096 --out $EXP/data/bpe4096`).

**The runs in `runs/` were made before `jaxley_pytorch.training` existed**, by this
folder's `train.py`, `rate_network.py` and `tasks.py` of commit 7e3a809, and were moved
into the configs' layout on 2026-10-25 (`runs/relayout_20261025.tsv` lists every old and
new name). Their directories hold those scripts' `result.json` and `teacher.pt` or
`student.pt`, but no `config.toml`: `configs/` holds their settings. The library
reproduces them exactly: for a few steps of the teacher and of students of every kind of
all three tasks, in float64 on the CPU, it gives the same training losses, evaluations
and parameters as those scripts, to the last bit (`probes/jaxley_pytorch/training_parity/`
on /nrs). (`runs/smoke/` holds the smoke runs of those scripts.)

`summarize.py` writes `summary.md` (tables), `training_curves.png` (Fig. 1c-e for every
task) and `outcomes.png` (every student's final task loss against its final activity
error); `traces.py` writes `traces_cycling.png` (Fig. 1f,g: the readout and two neurons in
the teacher and in the best student of each start). Figures: `$EXP/figures`.
Measured on an H100 (smoke test): a training step takes 0.085 s (cycling, 32 trials) or
0.55 s (text, 32 x 256 tokens, 14,800 tokens/s); a text student's evaluation takes about
22 s; at most 1.4 GB of GPU memory. Estimated: text teachers 4 h, text students 1.8 h.

## Stress test (cycling)

Can any training strategy that uses only what a student may know (the connectome, the
task, generic priors; never the teacher's activity or its trained gains and biases) reach
both the teacher's task loss and the teacher's activity? Target: task loss at most twice
the teacher's and activity error at most 0.1 of the shuffled baseline (the control
reaches 0.002). Ten students per strategy (`submit_stress.sh`; runs in
`runs/<config>/seed<k>`; `stress_test.py` writes `stress_test.md` and `stress_test.png`):

| strategy | config |
|---|---|
| random start, drawn again while its activity runs away; then 500 full-batch L-BFGS steps | `cycling_rstable_lbfgs.toml` |
| uniform start (gains 1, biases 0; gains scaled by 0.9 while the activity runs away); then L-BFGS | `cycling_uniform_lbfgs.toml` |
| random start, stable draws, trained on 6,400 new trials of the task instead of the 60 | `cycling_rstable_fresh.toml` |
| the shuffled-start students, continued with 500 L-BFGS steps | `cycling_shuffled_lbfgs.toml` |
| the 6,400-trials strategy, longer: 30,000 steps, batch 128, 1,000 warmup steps | `cycling_rstable_fresh_long.toml` |

## Results

**On the paper's cycling task, its result holds: students that learn the task do it with
single-neuron activity far from the teacher's. On next-token prediction (characters or
BPE tokens), it does not: every student that learns the task ends with essentially the
teacher's activity (per-neuron correlation 0.993-0.999), as the hypothesis predicted.**
Parameters are only partly recovered in either case: gains come closer to the teacher's,
biases do not.

Figures and tables are in
`/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity/figures`:

| file | what it shows |
|---|---|
| `activity_by_task.png` | every student's final activity error (relative to the shuffled-neuron baseline), per task |
| `training_curves.png` | the paper's Fig. 1c-e per task: task loss, activity, gain and bias errors during training |
| `outcomes.png` | final task loss against final activity error, per task |
| `traces_cycling.png`, `traces_chars.png`, `traces_tokens.png` | the paper's Fig. 1f,g / 5c: outputs and two example neurons, teacher vs students |
| `stress_test.png`, `stress_test.md` | the cycling stress test (below) |
| `summary.md`, `summary.json` | all numbers |

### Teachers

| task | validation loss | activity: participation ratio, components for 90% of variance | variance in the readout's directions |
|---|---|---|---|
| cycling | MSE 0.000126 | 1.9, 3 | 0.0001 (readout: 1 direction) |
| chars | 1.560 bits/char | 21.1, 85 | 0.14 (readout: 75 directions) |
| tokens | 3.606 bits/token = 0.907 bits/char | 25.0, 66 | 0.16 (readout: 75 directions) |

### Students (shuffled start, as in the paper; trained on the task itself)

Median [min, max]; errors are relative to the shuffled-neuron baseline (1 = no better than
the teacher's own neurons in random order).

| | cycling (10) | chars (10) | tokens (9; 1 diverged early) |
|---|---|---|---|
| task loss | 0.028 [0.00097, 0.051], teacher 0.000126 | 0.4-0.6% above the teacher's | 0.6-1.5% above the teacher's |
| output vs teacher | readout MSE 0.028 | KL 0.0065 bits/token | KL 0.031 bits/token |
| activity error, start -> end | 0.59 -> 100 [5.0, 612] | 0.30 -> 0.0015 [0.0014, 0.0019] | 0.44 -> 0.0047 [0.0042, 0.013] |
| per-neuron correlation with the teacher | 0.32 (shuffled neurons: 0.30) | 0.9987 | 0.9973 |
| ... in the readout's / other directions | 0.0057 / 100 | 0.0016 / 0.0015 | 0.0021 / 0.0053 |
| gain error | 1.32 | 0.23 | 0.31 |
| bias error | 5.8 | 0.95 | 1.27 |
| control (teacher's own start): activity error | 0.0024 | 0.00004 | 0.0001 |

* **Cycling reproduces the paper.** Students' activity moves away from the teacher's as
  they learn (Fig. 1d of the paper); all of the mismatch lies outside the one direction
  the readout sees. Students started from random gains and biases (drawn like the
  teacher's initial ones) end much closer (activity error 1.45 [0.37, 2.33]) and with lower
  task loss, but 4 of 10 such starts run away. Even students trained to within 1.5-1.8
  times the teacher's task loss keep activity errors of 0.41-1.19 (stress test below).
* **Text pins the activity.** Students reach the teacher's loss within 1.5% and its
  activity to within 0.14-1.4% of the shuffled baseline: 100 to 200 times closer than
  where they started. The pre-registered criterion (activity error <= 0.1 at the teacher's loss)
  is met by all 10 character students and all 9 token students.
* **Not just the readout's width.** The character teacher's activity needs 85 components
  for 90% of its variance, more than the 75 directions its readout sees, and only 14% of
  its variance lies in those directions; yet the students' error is as small outside them
  as inside (0.0015 vs 0.0016). For tokens (66 components) the readout's width alone could
  in principle account for it.
* **Parameters stay degenerate.** Biases end as far from the teacher's as shuffled values
  (they barely affect the activity); gains come 3 to 4 times closer. This matches the
  paper's stiff and sloppy parameter directions.

Caveats: with the paper's protocol the cycling students do not reach their teacher's task
loss while the text students do; but in the longest stress-test run three cycling students
come within 1.5-1.8 times the teacher's task loss and still have activity errors of
0.41-1.19 (below), so the cycling result does not rest on students failing the task. One
network size (300 neurons), one teacher per task. The text students' activity errors were
still falling at 10,000 steps. The cycling control ran 7,000 steps (the other cycling
students 15,000).

### Stress test (cycling)

No strategy reached the target (task loss at most twice the teacher's and activity error
at most 0.1): 0 of 70 students.

| strategy | task loss: median, best | activity error: median, best | ran away |
|---|---|---|---|
| shuffled start (the paper's) | 0.028, 0.00097 | 100, 5.0 | 0 |
| shuffled start + L-BFGS | unchanged | unchanged | 0 |
| random start | 0.0011, 0.00045 | 1.45, 0.37 | 4 |
| random start, stable draws + L-BFGS | 0.0012, 0.00045 | 1.25, 0.37 | 0 |
| uniform start + L-BFGS | 0.0014, 0.0013 | 4.4, 3.0 | 0 |
| random start, stable draws, 6,400 trials | 0.0014, 0.00049 | 1.24, 0.36 | 0 |
| ... with 30,000 steps, batch 128, 1,000 warmup steps | 0.00048, 0.00019 | 1.14, 0.41 | 0 |

With 15,000 steps every strategy stalls at 4-10 times the teacher's task loss. Training
the 6,400-trials students twice as long with four times the batch lowers their task loss
another 2-6 times (same starting draws): three come within twice the teacher's (1.49,
1.79 and 1.49 times), but their activity errors stay at 0.41, 0.60 and 1.19 of the
baseline (per-neuron correlation 0.36-0.63; 1 is the teacher). Across all ten, the
activity error hardly moves (median 1.24 -> 1.14) while the task loss falls: in this
regime, solving the task better does not bring the activity closer to the teacher's. The
uniform starts all reach the same solution, which is not the teacher's. L-BFGS never
accepted a step from where Adam stopped (gradient norms 1-850), so it cannot tell whether
the students are at local minima or just slow.
