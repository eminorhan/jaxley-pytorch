# Character-level language models of biophysical neurons

A prototype: recurrent networks of biophysical neuron models, simulated with
jaxley-pytorch, trained as character-level language models on TinyStories by
backpropagation through the simulator.

## Data

[karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
(revision `0397e27`, in the Hugging Face home `~/huggingface`): the GPT-4 stories of
TinyStories (Eldan & Li, 2023), cleaned to 74 ASCII characters. `prepare_data.py`
tokenizes it at the character level: the 74 characters and an end-of-story token, a
vocabulary of 75. The dataset's suggested splits give 2,712,634 training stories
(2.18 billion tokens), 10,000 validation and 10,000 test stories (8.0 and 8.1 million
tokens). The token files are in
`/nrs/scicompsoft/orhane/probes/jaxley_pytorch/language_modeling/data`.

```bash
python -I prepare_data.py --parquet ~/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/snapshots/0397e27157956705a0260709da3095bb9c43d6a7/tinystories_gpt4_clean.parquet --out <data dir>
```

## Models (`models.py`)

Every network has `n` neurons, 80% of them excitatory. Every neuron receives 100
synapses, 80 from random excitatory and 20 from random inhibitory neurons (no
autapses), and Dale's law holds: all synapses of a neuron have the sign of its type.
The trainable synaptic weights are `softplus(theta) * scale`, so they cannot change
sign.

Every character is presented for 5 ms of network time: a learned vector of input
currents (one per neuron, as a current density), constant over the 5 ms, is injected
into every neuron. The readout is a linear map from each neuron's voltage, averaged
over the 5 ms, to the 75 next-character logits. The network state carries over from
one character to the next.

| model | neuron | compartments per neuron | state variables per neuron | synapses | solver step |
|---|---|---|---|---|---|
| `rate` | passive membrane (leak, 10 ms time constant) | 1 | 1 | current-based, sigmoid of the presynaptic voltage | 0.5 ms |
| `hh` | Hodgkin-Huxley | 1 | 4 | conductance-based, first-order kinetics | 0.025 ms |
| `pospischil` | Pospischil et al. (2008): regular-spiking excitatory (Na, K, M-type K, leak), fast-spiking inhibitory (Na, K, leak) | 1 | 5 (exc.), 4 (inh.) | as `hh` | 0.025 ms |
| `l5pc` | the Jaxley paper's RNN unit (Fig. 4): apical, soma and basal compartments with L5PC channels; synapses from the soma onto the apical compartment, input into the basal one, readout from the soma | 3 | 20 | as `hh` | 0.025 ms |
| `gru` | a conventional GRU (dense, not biophysical), for reference | - | - | - | - |

The spiking models' synapses have reversal potentials of 0 mV (excitatory) and -80 mV
(inhibitory), a rise rate of 1/ms and decay time constants of 5 and 10 ms, and release
transmitter when the presynaptic voltage exceeds -20 mV (slope 2 mV), i.e. at spikes.
The input scales were set so that the untrained networks fire at 25-45 Hz when driven
by text (`train.py --mode activity`).

Two synapse models are the example's own:

* **Rate synapses.** Jaxley's `TanhRateSynapse` multiplies the weight by a tanh of the
  presynaptic voltage, which is negative below its offset, so an excitatory synapse
  would then inhibit. `SigmoidRateSynapse` uses a sigmoid (a non-negative rate)
  instead, so the weight's sign is the synapse's sign.
* **Spiking synapses.** `KineticSynapse` is the standard first-order kinetic model
  (Destexhe, Mainen & Sejnowski, 1994), with a bounded rise rate.

Two float32 problems of the library, inherited from Jaxley, surfaced here and have
since been fixed in `jaxley_pytorch` (`tests/test_float32_safety.py`): the HH and
Pospischil channels (and jaxley-mech's retinal ganglion cell channels) computed rates
`x / (exp(x) - 1)` that are 0/0 at one voltage each (e.g. -40 mV for HH's m gate),
which float32 voltages hit now and then (the NaN then spreads through the network);
and `IonotropicSynapse` computed its time constant as `(1 - s_inf) / k_minus`, which
is 0, with a NaN gradient, once a steep release saturates (`s_inf` rounds to 1 during
a spike). The results below were computed before that fix, with float32-safe copies of
the HH and Pospischil channels in the example; the library's channels now agree with
those copies to round-off, and the example uses them.

## Training (`train.py`)

Truncated backpropagation through time: 32 parallel streams through the training
text; every step simulates the next 32 characters (160 ms) of every stream from the
state the previous step ended in and backpropagates through those 32 characters.
Every character's simulation is checkpointed (recomputed in the backward pass). Adam
(learning rate 3e-3, the readout's scaled by 1024 / n for larger networks), the
gradient norm of every parameter tensor clipped at 1, float32. On the GPU, the whole
training step (simulation, loss, backward pass, Adam) is compiled (`torch.compile`)
and captured in a CUDA graph.

```bash
python train.py --mode train --model pospischil --n 1024 --hours 4 --eval-every 100 --out <dir>
python train.py --mode bench --model hh --n 4096 --out <dir>       # throughput only
python train.py --mode sample --out <dir of a training run>        # generate text
```

## Results

![Throughput and training curves](results.png)

### Throughput

One NVIDIA H100 (80 GB) unless noted, float32, 32 streams of 32 characters per step
(1,024 characters), 5 ms of network time per character (10 solver steps for `rate`,
200 for the spiking models); `train.py --mode bench`. Rows with batch 8 are the
largest networks, which only fit with 8 streams and checkpointing inside every
character's simulation. 100M characters are about 100,000 steps of 1,024 characters.

| model | neurons | synapses | solver step (ms) | batch | ms per solver step | characters / s | days per 100M characters | peak GPU memory (GB) | GPU |
|---|---|---|---|---|---|---|---|---|---|
| gru | 1,024 | - | - | 32 | - | 535,732 | 0.00216 | 0.4 | H100 |
| gru | 4,096 | - | - | 32 | - | 87,981 | 0.0132 | 2.9 | H100 |
| gru | 16,384 | - | - | 32 | - | 5,933 | 0.195 | 40.0 | H100 |
| rate | 1,024 | 102,400 | 0.5 | 32 | 0.73 | 4,372 | 0.265 | 1.5 | H100 |
| rate | 2,048 | 204,800 | 0.5 | 32 | 1.33 | 2,406 | 0.481 | 2.9 | H100 |
| rate | 4,096 | 409,600 | 0.5 | 32 | 2.48 | 1,292 | 0.896 | 5.6 | H100 |
| rate | 8,192 | 819,200 | 0.5 | 32 | 4.81 | 666 | 1.74 | 11.0 | H100 |
| rate | 16,384 | 1,638,400 | 0.5 | 32 | 10.43 | 307 | 3.77 | 22.0 | H100 |
| rate | 32,768 | 3,276,800 | 0.5 | 32 | 21.14 | 151 | 7.65 | 43.8 | H100 |
| rate | 65,536 | 6,553,600 | 0.5 | 8 | 10.35 | 77 | 15 | 73.8 | H100 |
| hh | 1,024 | 102,400 | 0.025 | 32 | 0.80 | 199 | 5.82 | 12.8 | H100 |
| hh | 2,048 | 204,800 | 0.025 | 32 | 1.44 | 111 | 10.4 | 24.6 | H100 |
| hh | 4,096 | 409,600 | 0.025 | 32 | 2.81 | 57 | 20.3 | 49.1 | H100 |
| hh | 8,192 | 819,200 | 0.025 | 8 | 1.63 | 25 | 47.1 | 55.4 | H100 |
| pospischil | 1,024 | 102,400 | 0.025 | 32 | 0.87 | 184 | 6.29 | 13.1 | H100 |
| pospischil | 2,048 | 204,800 | 0.025 | 32 | 1.50 | 106 | 10.9 | 25.1 | H100 |
| pospischil | 4,096 | 409,600 | 0.025 | 32 | 2.87 | 56 | 20.7 | 50.0 | H100 |
| pospischil | 8,192 | 819,200 | 0.025 | 8 | 1.69 | 24 | 48.9 | 55.9 | H100 |
| pospischil | 16,384 | 1,638,400 | 0.025 | 8 | 3.00 | 13 | 86.9 | 110.3 | H200 |
| l5pc | 1,024 | 102,400 | 0.025 | 32 | 1.00 | 160 | 7.23 | 15.0 | H100 |
| l5pc | 2,048 | 204,800 | 0.025 | 32 | 1.73 | 93 | 12.5 | 29.2 | H100 |
| l5pc | 4,096 | 409,600 | 0.025 | 32 | 3.23 | 50 | 23.3 | 58.3 | H100 |
| l5pc | 8,192 | 819,200 | 0.025 | 8 | 1.89 | 21 | 54.5 | 67.6 | H100 |

* The cost is set by the synapses, not the neuron model: all biophysical networks take
  0.7-1.0 ms per solver step at 1,024 neurons (100k synapses, 32 streams; forward,
  recomputation and backward pass together), i.e. about 4 billion synapse-steps per
  second, and the time grows linearly with the number of synapses. The
  three-compartment L5PC units with ten channel types cost only 25% more than
  Hodgkin-Huxley point neurons.
* Characters per second are therefore set by the solver steps per character: the rate
  model (0.5 ms steps) is 22-27x faster than the spiking models at the same size. With a
  0.05 ms step, the spiking models are 2x faster (e.g. Hodgkin-Huxley: 405, 227, 117
  and 55 characters/s at 1,024 to 8,192 neurons, 25 at 16,384), at a cost in accuracy
  (below).
* Largest networks that train on one H100: `rate` 65,536 neurons (6.6M synapses,
  batch 8); the spiking models 8,192 (batch 8; 16,384 with a 0.05 ms step), and
  Pospischil 16,384 at 0.025 ms on an H200 (141 GB). 131,072 rate neurons and 32,768
  spiking neurons ran out of memory in every configuration tried. Building a network
  takes about 18 ms per neuron (20 min for 65,536); compilation and graph capture
  20 s to 7 min.
* 100,000 steps of 1,024 characters in under a week: the rate model up to 16,384
  neurons (3.8 days; 32,768 take 7.7); Hodgkin-Huxley and Pospischil networks of
  1,024 neurons with the 0.025 ms step (5.8 and 6.3 days; L5PC 7.2), or of 2,048 with
  the 0.05 ms step (5.1 and 5.3 days).

### Training

Four hours per biophysical network, one hour per GRU, learning rate 3e-3 unless noted
(`train.py --mode train`, validation on 8 chunks of 32 characters in 32 streams):

| model | neurons | learning rate | parameters | characters trained | hours | characters / s | train bpc (last 100 steps) | validation bpc |
|---|---|---|---|---|---|---|---|---|
| gru | 192 | 0.003 | 0.25M | 2,398.2M | 1.0 | 1,016,079 | 1.68 | 1.75 |
| gru | 192 | 0.001 | 0.25M | 2,344.8M | 1.0 | 1,026,031 | 1.62 | 1.66 |
| gru | 192 | 0.0003 | 0.25M | 2,002.0M | 1.0 | 1,032,941 | 1.14 | 1.22 |
| gru | 1,024 | 0.003 | 6.45M | 1,585.3M | 1.0 | 535,399 | 1.65 | 1.79 |
| gru | 1,024 | 0.001 | 6.45M | 1,488.6M | 1.0 | 481,013 | 1.39 | 1.59 |
| gru | 1,024 | 0.0003 | 6.45M | 1,484.4M | 1.0 | 476,583 | 1.36 | 1.39 |
| rate | 1,024 | 0.003 | 0.26M | 61.0M | 4.0 | 4,372 | 2.04 | 2.11 |
| rate | 16,384 | 0.003 | 4.10M | 3.9M | 4.0 | 308 | 2.68 | 2.83 |
| hh | 1,024 | 0.003 | 0.26M | 2.9M | 4.0 | 206 | 3.24 | 3.19 |
| pospischil | 1,024 | 0.003 | 0.26M | 2.6M | 4.0 | 184 | 2.99 | 3.04 |
| pospischil | 4,096 | 0.003 | 1.02M | 0.7M | 4.0 | 55 | 2.81 | 2.84 |
| l5pc | 1,024 | 0.003 | 0.26M | 2.0M | 4.0 | 147 | 2.35 | 2.52 |

The same models after the same number of characters (training loss in bits per
character, mean over the logging window), i.e. what they learn from equal data:

| model | neurons | learning rate | 0.5M | 1M | 2M | 5M | 50M |
|---|---|---|---|---|---|---|---|
| gru | 192 | 0.003 | 1.85 | 1.71 | 1.57 | 1.49 | 1.67 |
| gru | 192 | 0.0003 | 2.55 | 2.16 | 1.84 | 1.61 | 1.33 |
| gru | 1,024 | 0.0003 | 1.79 | 1.59 | 1.40 | 1.28 | 1.09 |
| rate | 1,024 | 0.003 | 3.27 | 3.11 | 2.89 | 2.66 | 2.11 |
| rate | 16,384 | 0.003 | 3.21 | 3.03 | 2.82 | - | - |
| hh | 1,024 | 0.003 | 3.50 | 3.31 | 3.19 | - | - |
| pospischil | 1,024 | 0.003 | 3.12 | 3.03 | 2.98 | - | - |
| pospischil | 4,096 | 0.003 | 2.91 | - | - | - | - |
| l5pc | 1,024 | 0.003 | 2.56 | 2.40 | 2.35 | - | - |

* All biophysical networks learn. Per character of data, the three-compartment L5PC
  network learns fastest (2.40 bpc after 1M characters, vs 3.0-3.3 for the other
  networks of 1,024 neurons), and the 4,096-neuron Pospischil network beats the
  1,024-neuron one at a quarter of the data. They remain far behind a GRU of the same
  parameter count (192 units: 1.7-2.2 bpc after 1M characters).
* In four hours the rate network sees 20-30x more characters than the spiking ones
  and gets furthest (2.11 bpc). The learning rate was not tuned for the biophysical
  networks; for the GRUs, 3e-3 is too high beyond ~10M characters (the loss rises
  again) and 3e-4 is best.

Samples (`train.py --mode sample`, temperature 0.8, after the prompt "Once upon a
time"; the first 200 characters of the first sample):

* GRU, 192 units (1.22 bpc): ", there was a little girl named Lily. Lily loved to
  play with a fierce dog. One day, Max saw a little boy named Tom went for a walk.
  Sue was smart and she was fun."
* rate, 1,024 (2.11 bpc): ", there we rac happy and traic to t a bir dack and dad
  ally smiled and said, "Good ar undireatise. She quickly. They were very happy and
  toghe from the goo dis tow ar"
* L5PC, 1,024 (2.52 bpc): "rowhopprd. Tom s fe can sortsomed tore spiny wand."
  She was a bright upon the want the froplentith the watche friend. "I spicy shad for
  an I avery but lake"
* Pospischil, 4,096 (2.84 bpc): ", to she sa ha bis candi nte scaus au shi beg acked f
  and acm and ald natt heral Tim thery and ad and Maxce care une the ply as bire was"
* Hodgkin-Huxley, 1,024 (3.19 bpc): ", to srd ny tr t s d noioutsyareom tumes. Yhe nd.
  The playay. Tit wed tuttano and ane the bat ad toee toed Bent t erilas a rd he so"

### What it took to make the spiking networks train

* **The solver step.** One neuron, a noisy current for 1 s, spikes compared with a
  0.0125 ms reference (`diagnose/dt_accuracy.py` on /nrs):

  | model | step 0.1 ms | 0.05 ms | 0.025 ms |
  |---|---|---|---|
  | Hodgkin-Huxley: spikes (reference 44) | 38 | 62 | 44 |
  | Hodgkin-Huxley: median spike shift | 28 ms | 0.68 ms | 0.21 ms |
  | Pospischil RS: spikes (reference 33) | 33 | 33 | 33 |
  | Pospischil RS: median spike shift | 4.5 ms | 1.6 ms | 0.53 ms |

  (float32 and float64 gave the same spikes.) At 0.05 ms, the gradients through the
  recurrent spiking dynamics also grew far faster: the gradient norm of the input
  weights of the Pospischil network over 16 characters was 2.7e5 at 0.05 ms and 55 at
  0.025 ms (0.56 without recurrent synapses), and that network did not learn at all.
* **Clipping every parameter tensor's gradient on its own** (rather than all
  together), so that large gradients through the network dynamics do not scale down
  the readout's. After 150 steps of the Pospischil network of 1,024 neurons: 3.28 bpc
  (per tensor), 3.75 (all together, with an unstable stretch), 3.32 (per tensor, and
  the gradient through the state also truncated every 8 characters).
* **The readout's learning rate scaled by 1024 / n**: with Adam, a logit moves by
  about the learning rate times n per step, and unscaled networks of 4,096 neurons
  and more diverged.

### Next steps

* A fused GPU kernel for the synapses (gather the presynaptic voltages, update the
  synapse states, scatter the currents, forward and backward) would attack the actual
  bottleneck; a rough estimate of the memory traffic per synapse update suggests 2-4x.
* Tune the learning rate (and a schedule) for the biophysical networks; train longer
  and at the larger sizes.

All runs: `/nrs/scicompsoft/orhane/probes/jaxley_pytorch/language_modeling`
(`bench`, `bench_dt0025`, `bench_h200`: throughput; `train`, `train_gru_lr`: training
runs with `result.json`, `model.pt`, `samples.json`; `jobs`: logs; `diagnose`: the
NaN, gradient and solver-step probes). Tables: `summarize.py`; figure: `plot_results.py`.
