# Does the task pin down single-neuron activity in networks of spiking neurons?

`../task-complexity`, `../network-size` and `../sparsity` ask, with the rate network of
Beiran & Litwin-Kumar (2025), whether a task constrains single-neuron activity given the
connectome: students share a teacher's synaptic weights and learn only its single-neuron
parameters (gains and biases) from next-token prediction. This experiment asks the same
with biophysical neurons, on the same token task.

## Design

`jaxley_pytorch.training` with `[model] name = "pospischil"` or `"hh"`
(`jaxley_pytorch/training/spiking_network.py`), the neurons and synapses of
`examples/language_modeling`:

* **Neurons**: one compartment each. Pospischil: excitatory neurons regular spiking (Na,
  K, M-type K, leak), inhibitory ones fast spiking (Na, K, leak); Hodgkin-Huxley: Na, K,
  leak. 80% excitatory. Solver step 0.025 ms (larger steps shift spikes and make the
  gradients grow, see `examples/language_modeling`).
* **Synapses**: 150 per neuron (120 excitatory, 30 inhibitory), conductance-based
  kinetic synapses (reversal 0 mV / -80 mV, decay 5 ms / 10 ms, released by spikes),
  non-negative conductances, so Dale's law holds.
* **Initial coupling**: weaker than `examples/language_modeling`'s, at which both
  networks are chaotic at initialisation: a 0.3% change of every conductance and bias
  moves the activity by 0.6 (Pospischil) and 0.2 (Hodgkin-Huxley) of the shuffled-neuron
  baseline, and the gradient over 128 tokens overflows float32 or is 4e10 times the one
  over 8. Each model starts at the strongest coupling tried whose gradients stay bounded
  (`probes/coupling` and `probes/currents` in the output folder):
  * Pospischil: a quarter of the example's synaptic scales, input scale 4 instead of 2.
    The same change moves the activity by 0.002, the gradient over 128 tokens is 0.9
    times the one over 8, the neurons fire at 23 Hz, and a neuron's recurrent current is
    a quarter of its input current. At half the coupling, the gradient over 128 tokens is
    1,600 to 3 million times the one over 8 (input scales 3-5).
  * Hodgkin-Huxley: a tenth of the example's synaptic scales, its input scale 6. The
    same change moves the activity by 0.005, the gradient stops growing after 32 tokens
    (19, 1.0e3, 1.1e3 and 1.0e3 over 8, 32, 128 and 256 tokens), the neurons fire at
    45 Hz, and the recurrent current is a third of the input current. At a quarter of the
    coupling, the gradient over 128 tokens is 4e4 to 1e14 times the one over 8 (input
    scales 4-8).
* **Single-neuron parameters**: every neuron's maximal conductances (trained as
  logarithms) and a bias current: 5 per regular-spiking neuron, 4 per fast-spiking or
  Hodgkin-Huxley neuron. The teacher's conductances start around the defaults of its
  type (log-normal, standard deviation 0.25), its biases at 0.
* **Task**: the BPE tokens of `../task-complexity`, every token a constant input current
  for 5 ms (200 solver steps), one learned value per neuron (a standard recurrent language
  model's input); the logits are a linear readout of every neuron's voltage averaged over
  the token's window (centred at -60 mV, scaled by 20 mV), its bias starting at the log
  frequencies of the training tokens, without the RMSNorm the rate experiments read
  through (their rates are unbounded; voltages and firing rates are not). At step 1,000
  of the teacher recipe this readout beat a zero starting bias by 0.03 and 0.08
  bits/token and RMSNorm (with the same bias) by 0.08 and 0.09 (Pospischil, HH;
  `probes/readout_bias_3way/summary.md` in the output folder).
* **Teachers**: everything trained (synaptic weights, single-neuron parameters, input
  and readout): 5,000 steps of 32 streams x 256 tokens (41M tokens), AdamW without weight
  decay, every tensor's gradient clipped at 1.
* **Students** (ten per network, seeds 1-10): the teacher's single-neuron parameters,
  every parameter permuted within the excitatory and within the inhibitory neurons; only
  these trained, with the teacher's schedule (5,000 steps, peak learning rate 3e-3),
  gradient norm clipped at 2. One **control** per network starts from the teacher's own.
  (Until 2026-11-08 students and controls trained for 2,000 steps at a peak of 1e-3; see
  Results.)
* **Learning rate**: linear warmup over the first 5% of the steps, then linear decay to
  zero.
* **Measurements**: as in `../task-complexity` (task loss, KL divergence from the teacher,
  activity error, per-neuron correlation, an error for every single-neuron parameter,
  each next to its shuffled-neuron baseline), and every network's firing rate and the
  students' spike counts per token against the teacher's.

## Stages

Every run before 2026-11-01 (the smoke runs, the pilots and the first stage 2) read in and
out through 75 dimensions instead of the standard input and readout above; those runs
were removed. The choices they led to (the initial coupling, the learning rate, the
surrogate gradient) are kept; the numbers below come from them. Stage 2 was redone with
the standard input and readout (Results); its first redo (a zero starting readout bias) was
cancelled after about 1,000 steps and is in `superseded/20261101/` in the output folder.

0. Smoke: the 1,024-neuron configs of both models, 20 steps each (`submit_smoke.sh`).
1. Pilot (1,024 neurons, each model, `submit_pilot.sh`, configs in `configs/pilots`): the
   teacher at peak learning rates 3e-4, 1e-3 and 3e-3 for 2,000 steps each
   (`pilot_<model>_lr*_teacher.toml`), then three students and a control of the one with
   the lowest validation loss at the end, 1,000 steps each (`pilot_<model>_students.toml`,
   `pilot_<model>_control.toml`). Then a student at peak learning rates 3e-3 and 1e-2 for
   2,000 steps, from shuffled (`pilot_<model>_students_lr*.toml`, stopped halfway) and
   from random single-neuron parameters drawn like the teacher's initial ones
   (`pilot_<model>_random_lr*.toml`, stopped before they started): where its activity goes
   once it closes the task-loss gap. The teachers' parameters (synaptic weights included)
   moved no further in training than a random walk of Adam steps, so the Pospischil
   teacher at lr 3e-3 again with 64 and 128 streams per batch instead of 32
   (`pilot_pospischil_lr3e-3_teacher_b*.toml`): is the gradient noise of the batch the
   bottleneck? On an RTX PRO 6000 a step takes 14.0, 25.2 and 45.5 s at batch 32, 64 and
   128; 256 does not fit in its 96 GB (an H200 fits 256, at 106 s per step, but not 512).
   The batch was not the bottleneck: over the 256-token chunk, the exact gradient of the
   network's dynamics changes direction when the parameters change by 1e-6 of their size
   (in float64 too, and without synapses). Then the lr-3e-3
   teacher of each model with a surrogate gradient
   (`pilot_<model>_lr3e-3_teacher_surrogate0.1.toml`, `na_activation_gradient = 0.1`: in
   the backward pass, the sodium activation reads the voltage with derivative 0.1; the
   simulation is unchanged). On the initial Pospischil network, this gradient keeps its
   direction (cosine above 0.95) when the parameters change by 1e-3 of their size, and an
   Adam step of 1e-3 or 3e-3 along it lowers the loss 6 to 12 times as much as one along
   the exact gradient, on its batch and on another. The surrogate teachers ended at 4.39
   (Pospischil) and 4.50 (Hodgkin-Huxley) bits/token, against 5.53 and 5.82 with the exact
   gradient, and their training now lives in the dynamics: putting the synaptic weights
   back to their initial values costs 0.44 and 0.23 bits/token (the exact Pospischil
   teacher: 0.002), all but the readout 4.06 and 4.98 (1.83).
2. Full runs at 1,024 neurons, both models (`submit.sh`): the teacher with the surrogate
   gradient (`na_activation_gradient = 0.1`) at peak learning rate 3e-3 for 5,000 steps,
   then its ten students (shuffled start) and the control with the same schedule (first
   with 2,000 steps at 1e-3; redone on 2026-11-08). Then 4,096 neurons.

## Running

Outputs go to `/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/complex-neurons`
(`runs/<experiment_name>/seed<k>/`, `jobs/`; smoke runs under `runs/smoke/` and
`jobs/smoke/`). Jobs run on a snapshot of the repo
(`probes/jaxley_pytorch/tools/snapshot.sh`) with `~/myvenv`, one H100 each.

```bash
bash submit_smoke.sh  # per model, one H100: the 1,024-neuron configs with --smoke (20 steps)
bash submit_pilot.sh pospischil teachers  # stage 1: the three teachers, one H100 each
bash submit_pilot.sh pospischil students  # then the students and the control of the chosen one
QUEUE=gpu_rtx6000 bash submit_pilot.sh pospischil random_lr  # students from random starts, 2,000 steps
QUEUE=gpu_rtx6000 bash submit_pilot.sh pospischil teachers_batch  # the teacher at batch 64 and 128
QUEUE=gpu_rtx6000 bash submit_pilot.sh pospischil variant surrogate0.1  # the teacher, surrogate gradient
bash submit.sh  # stage 2: per model, the teacher, then its ten students and the control (RTX PRO 6000s)
```

## Results (stage 2)

**The spiking students learn the task and take on the teacher's activity, but not its
single-neuron parameters.** With the teacher's schedule their task loss ends 0.5%
(Pospischil) and 0.03% (Hodgkin-Huxley) above the teacher's, their activity error at 3.0%
and 1.8% of the shuffled-neuron baseline (the controls: 1.9% and 1.3%; the rate networks'
students: 0.02-0.06%), and their spike counts' correlation error at 0.32 and 0.15 (controls
0.22 and 0.12; shuffled 0.97). Their parameters end 0.17-0.78 (Pospischil) and 0.23-1.36
(Hodgkin-Huxley) of the baseline's error away from the teacher's, and the ten students of a
model end at nearly the same errors. The teacher's parameters are not where training stays
either: at an unchanged task loss, the controls' parameters move 0.14-0.64 (Pospischil) and
0.14-0.53 (Hodgkin-Huxley) away from them. Over the second half of training the students'
errors in gNa and gLeak were still falling (by 14-29%), those in gK, gKm and the biases
hardly (Pospischil: by less than 4%; Hodgkin-Huxley: gK by 9%, the biases by 1%).

Until 2026-11-08 students and controls trained for 2,000 steps at a peak learning rate of
1e-3, a third of the teacher's. Those students ended 11% and 16% above the teacher's task
loss with activity errors of 16% and 29%, still improving when the learning rate reached
zero: Adam moves a parameter by about the learning rate per step, about 1 unit over that
schedule, and the shuffled starts are 1.3-1.7 units from the teacher's values in gNa, gKm
and gLeak. They were rerun with `../rerun_students.sh`; the earlier runs are in
`superseded/20261108/` in the output folder, and `../cell-types/probes/student_budget`
there compares the schedules.

Figures and tables are in
`/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/complex-neurons/figures`
(`summarize.py`: `summary.md`, `training_curves.png`, `parameters.png`, `outcomes.png`,
`by_group.png`; `../task-complexity/traces.py --task tokens --name <model>_n1024 --configs configs`:
`traces_<model>_n1024.png`, the probability of the actual next token and two example
neurons' window-mean voltages, teacher vs the best student before and after training).

| model | teacher: task loss | firing rate | participation ratio | components for 90% |
|---|---|---|---|---|
| Pospischil | 3.695 bits/token | 23.4 Hz | 70.8 | 417 |
| Hodgkin-Huxley | 3.994 bits/token | 20.8 Hz | 51.4 | 477 |

Students (ten per model, shuffled start; median [min, max]; errors relative to the
shuffled-neuron baseline; the readout reads every direction of the activity):

| | Pospischil | Hodgkin-Huxley |
|---|---|---|
| task loss above the teacher's (%), start -> end | 33 -> 0.48 [0.27, 0.63] | 44 -> 0.03 [-0.02, 0.10] |
| KL from the teacher (bits/token) | 0.078 | 0.034 |
| activity error, start -> end | 0.27 -> 0.030 [0.029, 0.033] | 0.43 -> 0.018 [0.017, 0.019] |
| 1 - mean abs r (shuffled) | 0.036 (0.955) | 0.028 (0.95) |
| spike counts: 1 - correlation (shuffled) | 0.32 (0.97) | 0.15 (0.97) |
| gNa / gK / gKm / gLeak / bias error | 0.52 / 0.51 / 0.78 / 0.17 / 0.60 | 0.26 / 0.66 / - / 0.23 / 1.36 |
| firing rate (teacher's) | 18.6 Hz (23.4) | 20.5 Hz (20.8) |
| control: activity error | 0.019 | 0.013 |
| control: gNa / gK / gKm / gLeak / bias error | 0.26 / 0.19 / 0.64 / 0.14 / 0.36 | 0.35 / 0.42 / - / 0.14 / 0.53 |
| control: firing rate | 20.9 Hz | 21.6 Hz |

Against the first stage 2 (75-wide input and readout; numbers from its notes, the runs were
removed): the teachers are better (3.96 and 4.10 bits/token then) and fire about half as
fast (40 and 39 Hz then). Its students trained with the earlier schedule (2,000 steps at
1e-3).
