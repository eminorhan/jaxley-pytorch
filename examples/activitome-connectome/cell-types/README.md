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
teacher trained for 2,000 steps at 1e-3, and the same measurements. The differences:

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
