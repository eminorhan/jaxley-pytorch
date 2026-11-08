# How does solution degeneracy depend on network size?

`../task-complexity` asks whether next-token prediction on TinyStories pins down the
single-neuron activity of a 300-neuron network: students share the teacher's synaptic
weights and learn only their gains and biases from the task (on the paper's cycling task
they end far from the teacher's activity). This experiment repeats the token version
with networks of 600 and 900 neurons, 2 and 3 times as many; task-complexity's token runs
are its first size.

## Design

The protocol of `../task-complexity`'s token task, one config per arm in `configs/`
(`n<N>_teacher.toml`, `n<N>_students.toml`, `n<N>_control.toml`), run by
`jaxley_pytorch.training`, with one change: the number of neurons N. (The optimizer is
AdamW without weight decay, which is Adam's update.)

| neurons | excitatory | synapses (about N^2 / 2) | gains and biases per student |
|---|---|---|---|
| 300 (`../task-complexity`) | 210 | 45,206 | 600 |
| 600 | 420 | ~180,000 | 1,200 |
| 900 | 630 | ~405,000 | 1,800 |

* **Network**: the paper's rate network and initialisation at every size: 70% excitatory,
  every ordered pair connected with probability 0.5, weights 2.4 / sqrt(N) * normal + 1 / N
  (inhibitory ones x4), Dale's law, the paper's stability test; one teacher per size
  (seed 0).
* **Task**: the 4,096 BPE tokens of `../task-complexity` (its `data/bpe4096`), every token a
  constant input current for 10 solver steps (tau / 2), with a standard recurrent
  language model's input and readout: one learned input value per neuron and token, and
  a linear readout of every neuron's rate after RMSNorm over the neurons
  (`readout_norm = "rms"`, as in `../task-complexity`), its bias starting at the log
  frequencies of the training tokens (`../sparsity`). (Until 2026-11-01 input and
  readout went through 75 dimensions; those runs were removed.)
* **Teachers**: recurrent weights, gains, biases, input and readout maps trained together:
  25,000 steps of 32 streams x 256 tokens (205M tokens), Adam at a peak learning rate
  of 3e-3, every tensor's gradient clipped at 1.
* **Students** (ten per size, seeds 1-10): the teacher's gains and biases permuted within
  the excitatory and within the inhibitory neurons, gains halved; only these trained, on
  the next token (cross-entropy): 10,000 steps (82M tokens), Adam at a peak learning
  rate of 1e-3, gradient norm clipped at 2. One **control student** per size starts from
  the teacher's own gains and biases.
* **Learning rate**, every run: linear warmup over the first 5% of the steps (teachers
  1,250, students 500), then linear decay to zero.
* **Measurements**: as in `../task-complexity`, on 64 validation segments of 256 tokens
  (after 32 tokens from rest): task loss, KL divergence from the teacher, activity error,
  per-neuron correlation, gain and bias errors, each next to its shuffled-neuron
  baseline; the teacher's activity dimensionality.

## Reading the results

R = (activity error after training) / (shuffled-neuron baseline), median over a size's ten
students; a student solves the task if its loss is within 2% of the teacher's. As in
`../task-complexity`, the task pins down the activity at a size if its students solve it
with R <= 0.1, and leaves it degenerate if R >= 0.5. The answer is how R, the per-neuron
correlation and the gain and bias errors change with N.

`../sparsity` repeats this experiment with 150 inputs per neuron at every size, which
separates the size from the fan-in.

## Running

Outputs go to `/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/network-size`
(`runs/<experiment_name>/seed<k>/`, `jobs/`, `figures/`; smoke runs under `runs/smoke/` and
`jobs/smoke/`). Jobs run on a snapshot of the repo
(`probes/jaxley_pytorch/tools/snapshot.sh`) with `~/myvenv`, one RTX PRO 6000 each (faster than
an H100 for these float32 networks).

```bash
bash submit_smoke.sh  # per size, one GPU: every config with --smoke (20 steps); then the summary
bash submit.sh        # per size: the teacher, then 10 students and a control; then the summary
```

Every run saves a checkpoint every 1,000 steps; a run that stopped early continues with
its command plus `--resume`. `summarize.py` reads these runs and, as the 300-neuron size,
`../task-complexity`'s token runs (`--task-complexity`, its runs directory); it writes
`summary.md` and `summary.json` (tables), `training_curves.png` (the paper's Fig. 1c-e, one row per size), `outcomes.png`
(every student's final task loss against its final activity error) and
`activity_by_size.png` (every student's final activity error against the size).

## Results

**At every size, students that learn the task take on the teacher's activity, and their
gains and biases are recovered less well the larger the network.** The trained students'
activity error is 0.4-0.5% of the shuffled-neuron baseline at 300, 600 and 900 neurons;
their gain error rises from 0.55 to 0.72 of the baseline, and their biases end further from
the teacher's than shuffled ones (1.66-1.95). Figures and tables are in
`/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/network-size/figures`
(also `traces_n600.png` and `traces_n900.png`: 64 tokens of validation text, the probability
of the actual next token and two example neurons' rates, teacher vs the best student before
and after training, made with `../task-complexity/traces.py`).

| neurons | synapses | teacher: task loss | participation ratio | components for 90% | mean rate |
|---|---|---|---|---|---|
| 300 | 45,206 | 3.479 bits/token | 33.4 | 120 | 0.343 |
| 600 | 180,409 | 3.231 bits/token | 23.3 | 190 | 0.271 |
| 900 | 405,148 | 3.084 bits/token | 18.9 | 245 | 0.225 |

Students (ten per size, shuffled start; median [min, max]; errors relative to the
shuffled-neuron baseline, for 1 - mean abs r 0.89-0.91; the readout reads every direction of
the activity):

| neurons | solved | task loss above the teacher's (%) | activity error, start -> end | 1 - mean abs r | gain error | bias error | control: activity error |
|---|---|---|---|---|---|---|---|
| 300 | 10 of 10 | 0.48 [0.41, 0.58] | 0.51 -> 0.0038 [0.0034, 0.0039] | 0.0018 | 0.55 | 1.66 | 4.1e-5 |
| 600 | 10 of 10 | 0.56 [0.49, 1.31] | 0.39 -> 0.0048 [0.0043, 0.0095] | 0.0038 | 0.70 | 1.95 | 8.4e-5 |
| 900 | 10 of 10 | 0.88 [0.83, 1.10] | 0.36 -> 0.0036 [0.0033, 0.0041] | 0.0050 | 0.72 | 1.94 | 5.5e-5 |
