# How does solution degeneracy depend on network size at a fixed fan-in?

In `../network-size`, the networks grow at the paper's connection probability of 0.5, so
a neuron's number of inputs grows with the network: 150, 300 and 450 inputs at 300, 600
and 900 neurons. This experiment separates the size from the fan-in: every neuron has 150
inputs on average at every size, so the connection probability falls as the network
grows.

## Design

The protocol of `../network-size` (the token task and recipe of `../task-complexity`), one
config per arm in `configs/`, with `fan_in = 150` in place of `connection_prob = 0.5`:

| neurons | connection probability | inputs per neuron | synapses | gains and biases per student |
|---|---|---|---|---|
| 300 (`../task-complexity`) | 0.5 | 150 | 45,206 | 600 |
| 600 | 0.25 | 150 | ~90,000 | 1,200 |
| 900 | 0.17 | 150 | ~135,000 | 1,800 |

* **Network**: the paper's rate network and initialisation, with every ordered pair
  connected with probability 150 / N (a neuron's number of inputs is binomial around
  150, as in the paper's networks) and the paper's weights scaled for 300 neurons
  instead of N: 2.4 / sqrt(300) * normal + 1 / 300 (inhibitory ones x4), so that a
  neuron's summed input has the statistics it has in the 300-neuron network. At 300
  neurons this draws `../task-complexity`'s network exactly, so its token runs are the
  first size.
* **Everything else** as in `../network-size`: the task, the teachers (25,000 steps, Adam
  at a peak learning rate of 3e-3), ten students per size with the teacher's schedule
  (until 2026-11-08: 10,000 steps at a peak of 1e-3; those runs are in
  `superseded/20261108/` in the output folder), the control, the measurements.

## The 900-neuron teacher and the readout bias

With the readout bias starting at zero, the 900-neuron teacher's rates ran away within
its first 31 training steps, RMSNorm and all. The network starts close to its stability
edge: the largest real part of the eigenvalues of its linearised dynamics is 0.87 at rest
(the paper's test accepts up to 1), up to 1.02 under the task's input, and 3.1 with every
neuron on the linear part of softplus, so nothing stops a mode that starts to grow. At
the start of training, the largest gain in the loss is the logits' input-independent part
(the token frequencies). The readout bias learns it slowly (Adam moves it by about the
learning rate per step); the recurrent weights learn it faster, by building
self-sustained activity, which raised the leading eigenvalue from 0.87 to 0.95 in 30
steps. Whether this crosses the edge depends on the random input and readout weights: 2
of 5 draws for this network ran away, 0 of 5 with a 75-wide input. The readout bias now
starts at the log frequencies of the training tokens (`Text.init_io`): the eigenvalue then
falls from the first step (0.87 to 0.82 at step 30 and 0.68 at step 120), and both draws
that ran away train. Every rate experiment's runs were redone with it on 2026-11-01 (the
runs with a zero bias are in `superseded/20261101/` in the output folder). Probes:
`probes/n900_teacher_nan/` in the output folder.

## Reading the results

Compare with `../network-size` at the same number of neurons. There the fan-in grows
with the network (300 inputs at 600 neurons, 450 at 900); here it stays at 150. A
difference between the two series at the same size is the effect of the fan-in, and the
trend within this series is the effect of the size at a fixed fan-in.

## Running

Outputs go to `/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/sparsity`
(`runs/<experiment_name>/seed<k>/`, `jobs/`, `figures/`; smoke runs under `runs/smoke/` and
`jobs/smoke/`). Jobs run on a snapshot of the repo
(`probes/jaxley_pytorch/tools/snapshot.sh`) with `~/myvenv`, one GPU each. The tables and
figures come from `../network-size/summarize.py`, with `../task-complexity`'s token runs as
the 300-neuron size.

```bash
bash submit_smoke.sh  # per size, one H100: every config with --smoke (20 steps); then the summary
bash submit.sh        # per size, one H100: the teacher, then 10 students and a control; then the summary
```

## Results

**At a fixed fan-in of 150 the pattern is that of `../network-size`: the students take on
the teacher's activity at every size, and their gains and biases are recovered less well
at 600 and 900 neurons than at 300.** Their gains end where those of the dense networks of
the same size do (gain error 0.41 and 0.39 of the baseline at 600 and 900 neurons in both
series); their biases end a little closer to the teacher's than in the dense networks
(1.55 and 1.52, against 1.79 and 1.81), still further than shuffled ones. With the earlier
schedule (10,000 steps at 1e-3) the gain errors were 0.64 and 0.66 and the bias errors
1.90 and 1.76. Figures and
tables are in `/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/sparsity/figures`
(also `traces_n600.png` and `traces_n900.png`, as in `../network-size`).

| neurons | synapses | teacher: task loss | participation ratio | components for 90% | mean rate |
|---|---|---|---|---|---|
| 300 | 45,206 | 3.479 bits/token | 33.4 | 120 | 0.343 |
| 600 | 90,060 | 3.276 bits/token | 25.5 | 192 | 0.304 |
| 900 | 134,892 | 3.161 bits/token | 21.8 | 246 | 0.288 |

Students (ten per size, shuffled start; median [min, max]; errors relative to the
shuffled-neuron baseline, for 1 - mean abs r 0.89-0.91):

| neurons | solved | task loss above the teacher's (%) | activity error, start -> end | 1 - mean abs r | gain error | bias error | control: activity error | control: gain / bias error |
|---|---|---|---|---|---|---|---|---|
| 300 | 10 of 10 | -0.02 [-0.03, -0.00] | 0.51 -> 0.00048 [0.00045, 0.00053] | 0.00033 | 0.23 | 1.02 | 1.6e-4 | 0.15 / 0.43 |
| 600 | 10 of 10 | -0.02 [-0.04, 0.30] | 0.40 -> 0.00064 [0.00063, 0.0025] | 0.00058 | 0.41 | 1.55 | 2.4e-4 | 0.17 / 0.64 |
| 900 | 10 of 10 | -0.06 [-0.07, 0.43] | 0.40 -> 0.00034 [0.00033, 0.0018] | 0.00075 | 0.39 | 1.52 | 1.0e-4 | 0.16 / 0.53 |
