# Does sharing single-neuron parameters within cell types make them identifiable?

In `../complex-neurons`, every Pospischil neuron has its own single-neuron parameters (5
per excitatory neuron, 4 per inhibitory one: 4,915 for 1,024 neurons), and students that
share the teacher's synapses learn the task only partly and recover neither its activity
nor its parameters. Here the neurons belong to a few discrete cell types that share their
single-neuron parameters, and the students know every neuron's type but not the types'
values: 18, 36 and 72 values to find with 4, 8 and 16 types. Does this reduction of the
degeneracy make the parameters identifiable, and the activity recoverable?

## Design

Everything as `../complex-neurons`' Pospischil stage 2: 1,024 neurons with 150 inputs each,
the BPE token task, the surrogate gradient (`na_activation_gradient = 0.1`), the teacher
trained for 5,000 steps at a peak learning rate of 3e-3, ten students and a control per
teacher trained with the teacher's schedule (until 2026-11-08: 2,000 steps at 1e-3, see
Results), and the same measurements. The differences:

* **Cell types** (`[model] cell_types = K`, K = 4, 8, 16): the 819 excitatory neurons in
  K / 2 contiguous groups of (nearly) equal size, the 205 inhibitory ones likewise (a
  type's neurons are random ones in the connectome, which is drawn at random). Every type
  has one value of each single-neuron parameter, which its neurons share: gNa, gK, gKm
  (excitatory types), gLeak and the bias. The teacher's values are drawn per type as a
  neuron's are in `../complex-neurons` (log-normal around the defaults of the class,
  standard deviation 0.25; biases 0), then trained with everything else.

  | types | excitatory types | neurons per excitatory type | inhibitory types | neurons per inhibitory type | values per network |
  |---|---|---|---|---|---|
  | 4 | 2 | 409-410 | 2 | 102-103 | 18 |
  | 8 | 4 | 204-205 | 4 | 51-52 | 36 |
  | 16 | 8 | 102-103 | 8 | 25-26 | 72 |

* **Students** (seeds 1-10): the teacher's synaptic weights, input and readout, and every
  neuron's type; they learn the types' values, starting from the teacher's values
  reassigned among the types of the same class so that no type keeps its own (each
  parameter separately). The **control** starts from the teacher's own values.
* **Reference**: `../complex-neurons`' Pospischil runs, in which every neuron is its own
  type (1,024 types; students start from the teacher's values shuffled among the neurons
  of the same class).

The errors of the single-neuron parameters are taken over the neurons (each type weighted
by its size), against the same shuffled-neuron baseline as in `../complex-neurons` (the
teacher's own values against a random permutation of its neurons).

## Reading the results

If sharing makes the parameters identifiable, the students' parameter errors fall far
below the shuffled-neuron baseline, towards the control's, the more so the fewer the
types. If the activity then follows, the students' activity error falls with them. If the
students instead solve the task with type values other than the teacher's, the degeneracy
does not come from every neuron having its own parameters alone.

## Running

Outputs go to `/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/cell-types`
(`runs/<experiment_name>/seed<k>/`, `jobs/`; smoke runs under `runs/smoke/` and
`jobs/smoke/`). Jobs run on a snapshot of the repo
(`probes/jaxley_pytorch/tools/snapshot.sh`) with `~/myvenv`, one RTX PRO 6000 each.

```bash
bash submit_smoke.sh  # per number of types: the teacher, one student and the control, 20 steps each
bash submit.sh        # per number of types: the teacher, then its 10 students and the control; then the summary
```

The summary is `../complex-neurons/summarize.py` with one `--group` per number of types and
`../complex-neurons`' Pospischil runs as the per-neuron reference: `summary.md` (tables),
`training_curves.png`, `parameters.png`, `outcomes.png` and `by_group.png` (every student's
final error in the activity and in every single-neuron parameter, per number of types).

`submit_lr5e-3.sh` and `configs/*_lr5e-3.toml` are the pilot at a peak learning rate of 5e-3
(Results); `../rerun_students.sh` reran every experiment's students and controls with the
teachers' schedules on 2026-11-08.

## Results

**Sharing the single-neuron parameters within cell types lets the students recover them:
all of them but gKm with 4 types, most with 16, part with 8. Without types
(`../complex-neurons`) the students reach the teacher's task loss and activity with other
parameters.** Tables and figures are in
`/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/cell-types/figures`.

| types | teacher: task loss | firing rate | participation ratio | components for 90% |
|---|---|---|---|---|
| 4 | 3.649 bits/token | 5.6 Hz | 75.5 | 396 |
| 8 | 3.615 bits/token | 14.2 Hz | 72.3 | 365 |
| 16 | 3.616 bits/token | 10.3 Hz | 71.4 | 363 |
| 1,024 (`../complex-neurons`) | 3.695 bits/token | 23.4 Hz | 70.8 | 417 |

Students (ten per teacher; median [min, max], without a range where the ten agree to the
digits shown; errors relative to the shuffled-neuron baseline):

| | 4 types | 8 types | 16 types | 1,024 types |
|---|---|---|---|---|
| task loss above the teacher's (%), start -> end | 36 -> -0.01 | 66 -> 1.41 [0.36, 2.66] | 38 -> 0.19 [0.01, 1.02] | 33 -> 0.48 [0.27, 0.63] |
| activity error, start -> end | 0.32 -> 0.0003 | 0.45 -> 0.011 [0.005, 0.019] | 0.36 -> 0.003 [0.001, 0.010] | 0.27 -> 0.030 [0.029, 0.033] |
| spike counts: 1 - correlation | 0.03 | 0.29 [0.15, 0.38] | 0.17 [0.07, 0.32] | 0.32 |
| firing rate (teacher's) | 5.7 Hz (5.6) | 11.9 Hz [8.9, 15.1] (14.2) | 9.4 Hz [7.8, 10.6] (10.3) | 18.6 Hz (23.4) |
| gNa error | 0.03 | 0.46 [0.18, 0.95] | 0.20 [0.08, 0.63] | 0.52 |
| gK error | 0.06 | 0.46 [0.22, 0.90] | 0.23 [0.10, 0.63] | 0.51 |
| gLeak error | 0.01 | 0.05 [0.02, 0.15] | 0.05 [0.02, 0.07] | 0.17 |
| bias error | 0.05 | 0.24 [0.09, 0.44] | 0.08 [0.04, 0.12] | 0.60 |
| gKm error | 0.57 | 0.71 [0.40, 1.03] | 1.01 [0.65, 1.43] | 0.78 |
| control: activity error | 0.0006 | 0.002 | 0.0015 | 0.019 |
| control: gNa / gK / gLeak / bias / gKm error | 0.01 / 0.11 / 0.02 / 0.05 / 1.72 | 0.10 / 0.10 / 0.04 / 0.15 / 0.78 | 0.03 / 0.08 / 0.05 / 0.06 / 1.35 | 0.26 / 0.19 / 0.14 / 0.36 / 0.64 |

* **4 types**: every student ends at the teacher's task loss, activity and spikes, with
  gNa, gK, gLeak and the biases back at the teacher's values (errors 0.01-0.06, the
  control's level). The ten students share one start (with two types per class, the swap
  is the only reassignment in which no type keeps its own values), so they differ only in
  the order of the training text.
* **16 types**: the students come most of the way. The three that end at the teacher's
  task loss (within 0.02%) have gNa errors of 0.08-0.13 and gK errors of 0.10-0.20; the
  three furthest from it (0.6-1.0% above) have 0.35-0.63 and 0.21-0.63.
* **8 types**: the students end at different values (gNa error 0.18-0.95), 0.4-2.7% above
  the teacher's task loss, and half of them fire at 9-10.5 Hz against the teacher's 14.2.
  Why 8 types do worse than 16 was not measured (their starts are worse: 5.3-6.9
  bits/token, against 4.9-5.3 with 16 types).
* In 8 and 16 types the students with the larger parameter errors also end with the
  higher task loss, and the errors were still falling when the learning rate reached zero
  (gNa and gK by 20-32% over the second half of training). This points to unfinished
  learning, not to other solutions at the teacher's task loss.
* **gKm is not pinned by the task**: the controls' gKm moves away from the teacher's (to
  0.78-1.72 of the baseline's error) while their task loss and activity stay; the
  students' ends at 0.57-1.01.
* **Without types**, the students reach the teacher's task loss within 0.5% and its
  activity within 3% of the baseline, with parameters 0.17-0.78 away from the teacher's,
  and the control's parameters leave the teacher's too (0.14-0.64).

Earlier rounds: until 2026-11-08 students and controls trained for 2,000 steps at a peak
learning rate of 1e-3, a third of the teacher's; near their end (step 1,900, before they
were cancelled) the students were 13-36% above the teacher's task loss, with activity
errors of 0.16-0.34. Adam moves a value by about the learning rate per step, about 1 unit
over that schedule, and the starts are 0.8-3.5 units from the teacher's values. A pilot at
5e-3 (one student per group, cancelled after step 1,300) followed the 1e-3 students' path
at the same summed learning rate, only faster, and the task loss falls all the way along
the straight line from a student's start to the teacher for 11 of 12 starts
(`probes/student_progress`, `probes/line_scan` and `probes/student_budget` in the output
folder; the 1e-3 runs are in `superseded/20261108/`, the pilot's in `runs/*_lr5e-3`).
