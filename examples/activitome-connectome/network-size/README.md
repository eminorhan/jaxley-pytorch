# How does solution degeneracy depend on network size?

In `../task-complexity`, next-token prediction on TinyStories pinned down the
single-neuron activity of a 300-neuron network: students that share the teacher's
synaptic weights and learn only their gains and biases from the task end with the
teacher's activity (BPE tokens: activity error 0.0047 of the shuffled-neuron baseline,
per-neuron correlation 0.997), while on the paper's cycling task they do not. This
experiment repeats the token version with networks of 512, 1,024 and 2,048 neurons.

## Design

The protocol and code of `../task-complexity` (`train.py --task tokens --n <N>`), token
task only; only the number of neurons N changes.

| neurons | excitatory | synapses (about N^2 / 2) | gains and biases per student |
|---|---|---|---|
| 300 (`../task-complexity`) | 210 | 45,206 | 600 |
| 512 | 358 | ~131,000 | 1,024 |
| 1,024 | 716 | ~524,000 | 2,048 |
| 2,048 | 1,433 | ~2,100,000 | 4,096 |

* **Network**: the paper's rate network and initialisation at every size: 70% excitatory,
  every ordered pair connected with probability 0.5, weights 2.4 / sqrt(N) * normal + 1 / N
  (inhibitory ones x4), Dale's law, the paper's stability test; one teacher per size
  (seed 0).
* **Task**: the 4,096 BPE tokens of `../task-complexity` (its `data/bpe4096`), every token a
  constant input current for 10 solver steps (tau / 2). Input and readout go through 75
  dimensions at every size, so the readout's directions are a smaller share of the
  activity space as N grows.
* **Teachers**: recurrent weights, gains, biases, input and readout maps trained together:
  25,000 steps of 32 streams x 256 tokens (205M tokens), Adam 3e-3, every tensor's
  gradient clipped at 1.
* **Students** (ten per size, seeds 1-10): the teacher's gains and biases permuted within
  the excitatory and within the inhibitory neurons, gains halved; only these trained, on
  the next token (cross-entropy): 10,000 steps (82M tokens), Adam 1e-3, gradient norm
  clipped at 2. One **control student** per size starts from the teacher's own gains and
  biases.
* **Learning rate**, every run: linear warmup over the first 5% of the steps (teachers
  1,250, students 500), then linear decay to zero.
* **Measurements**: as in `../task-complexity`, on 64 validation segments of 256 tokens
  (after 32 tokens from rest): task loss, KL divergence from the teacher, activity error
  (also split into the readout's 75 directions and the rest), per-neuron correlation,
  gain and bias errors, each next to its shuffled-neuron baseline; the teacher's activity
  dimensionality.

## Reading the results

R = (activity error after training) / (shuffled-neuron baseline), median over a size's ten
students; a student solves the task if its loss is within 2% of the teacher's. As in
`../task-complexity`, the task pins down the activity at a size if its students solve it
with R <= 0.1, and leaves it degenerate if R >= 0.5. The answer is how R, the per-neuron
correlation and the split of the error between the readout's directions and the rest
change with N.

## Running

Outputs go to `/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/network-size`
(`runs/n<N>/{teacher,student_seed<k>,student_control}/`, `jobs/`, `figures/`; smoke runs
under `runs/smoke/` and `jobs/smoke/`). Jobs run on a snapshot of the repo
(`probes/jaxley_pytorch/tools/snapshot.sh`) with `~/myvenv`, one H100 each.

```bash
bash submit_smoke.sh  # per size, one H100: 20 steps of a teacher, a student and a control, and the equivalence check; then the summary
bash submit.sh        # per size: the teacher, then 10 students and a control; then the summary
```

`summarize.py` writes `summary.md` and `summary.json` (tables), `training_curves.png` (the
paper's Fig. 1c-e, one row per size), `outcomes.png` (every student's final task loss
against its final activity error) and `activity_by_size.png` (every student's final
activity error, one column per size).
