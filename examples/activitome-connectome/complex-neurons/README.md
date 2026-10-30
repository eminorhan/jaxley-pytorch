# Does the task pin down single-neuron activity in networks of spiking neurons?

`../task-complexity`, `../network-size` and `../sparsity` ask, with the rate network of
Beiran & Litwin-Kumar (2025), whether a task constrains single-neuron activity given the
connectome: students that share a teacher's synaptic weights and learn only its
single-neuron parameters (gains and biases) recover the teacher's activity on
next-token prediction, at every size and fan-in tried, but not its gains and biases.
This experiment asks the same with biophysical neurons, on the same token task.

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
  for 5 ms (200 solver steps); the readout reads every neuron's voltage averaged over the
  token's window (centred at -60 mV, scaled by 20 mV).
* **Teachers**: everything trained (synaptic weights, single-neuron parameters, input
  and readout): 5,000 steps of 32 streams x 256 tokens (41M tokens), AdamW without weight
  decay, every tensor's gradient clipped at 1.
* **Students** (ten per network, seeds 1-10): the teacher's single-neuron parameters,
  every parameter permuted within the excitatory and within the inhibitory neurons; only
  these trained, 2,000 steps (16M tokens), peak learning rate 1e-3, gradient norm clipped
  at 2. One **control** per network starts from the teacher's own.
* **Learning rate**: linear warmup over the first 5% of the steps, then linear decay to
  zero.
* **Measurements**: as in `../task-complexity` (task loss, KL divergence from the teacher,
  activity error split into the readout's directions and the rest, per-neuron
  correlation, an error for every single-neuron parameter, each next to its
  shuffled-neuron baseline), and every network's firing rate and the students' spike
  counts per token against the teacher's.

## Stages

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
   the exact gradient, on its batch and on another.
2. Full runs: Pospischil at 1,024 and 4,096 neurons, then Hodgkin-Huxley.

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
```
