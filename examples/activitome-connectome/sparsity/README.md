# How does solution degeneracy depend on network size at a fixed fan-in?

In `../network-size`, the networks grow at the paper's connection probability of 0.5, so
a neuron's number of inputs grows with the network: 150, 300 and 450 inputs at 300, 600
and 900 neurons. There, students recover the teacher's activity at every size, while
their gains and biases get less like the teacher's as the network grows. This experiment
separates the size from the fan-in: every neuron has 150 inputs on average at every size,
so the connection probability falls as the network grows.

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
  at a peak learning rate of 3e-3), ten students per size (10,000 steps, peak 1e-3), the
  control, the measurements.

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
