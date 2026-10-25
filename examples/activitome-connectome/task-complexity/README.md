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

* **Network** (`rate_network.py`): the paper's rate network in jaxley_pytorch. 300
  neurons (70% excitatory), each a passive compartment (time constant tau = 10 ms) whose
  rate is `relu(g) * softplus(x + b)`, current-based softplus synapses, every ordered pair
  connected with probability 0.5, Dale's law, the paper's initialisation and stability
  test. Forward Euler with dt = 0.5 ms is exactly the update of the paper's code
  (alpha = 0.05); `train.py --role check` verifies this against a direct implementation
  of that update (rates and gradients agree to 1e-15 in float64).
* **Tasks** (`tasks.py`):
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
* **Teachers** (`train.py --role teacher`): recurrent weights, gains, biases and the
  readout trained together (and, for text, the input map); the cycling task's input
  weights stay random, as in the paper. Cycling: the paper's recipe (Adam 1e-3, batches
  of 32 of the 60 trials), 3,000 steps instead of 1,400. Text: 32 streams through the
  training text, backpropagation through windows of 256 tokens (the state carries over
  from window to window), Adam 3e-3 with every tensor's gradient clipped at 1 (as in
  `examples/language_modeling`), 25,000 steps (205M tokens). An average story is 803
  characters or 202 BPE tokens.
* **Students** (`train.py --role student`, ten per task): the teacher's weights, inputs,
  readout and initial state; gains and biases initialised as in the paper's code (the
  teacher's, permuted within the excitatory and within the inhibitory neurons, gains
  halved), and only these trained (Adam 1e-3, gradient norm clipped at 2) on the task
  itself: the cycling task's target (mean squared error, 7,000 steps, as in the paper) or
  the next token of the text (cross-entropy, 10,000 steps of 32 x 256 tokens). The
  teacher's outputs are not used. One **control student** per task starts from the
  teacher's own gains and biases: how far training alone moves a student away from the
  teacher sets the floor of the activity error.
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

Outputs go to `/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/task-complexity`
(`data/`, `runs/<task>/{teacher,student_seed<k>,student_control}/`, `jobs/`, `figures/`;
smoke runs under `runs/smoke/` and `jobs/smoke/`). Jobs run on a snapshot of the repo
(`probes/jaxley_pytorch/tools/snapshot.sh`) with `~/myvenv`.

```bash
bash submit_smoke.sh  # BPE token files (CPU job), then smoke.sh on one H100
bash submit.sh        # 3 teachers, 3 x 10 students + 3 controls (each after its teacher), summary
```

`summarize.py` writes `summary.md` (tables) and `fig1.png` (Fig. 1c-e for every task).
Measured on an H100 (smoke test): a training step takes 0.085 s (cycling, 32 trials) or
0.55 s (text, 32 x 256 tokens, 14,800 tokens/s); a text student's evaluation takes about
22 s; at most 1.4 GB of GPU memory. Estimated: text teachers 4 h, text students 1.8 h.

## Results

Not run yet.
