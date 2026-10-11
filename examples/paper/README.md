# Reproducing the Jaxley paper with jaxley-pytorch

Scripts that redo the experiments of Deistler et al., *Jaxley: differentiable simulation
enables large-scale training of detailed biophysical models of neural dynamics*
(Nature Methods 22, 2649–2657, 2025) with this port. Each script is a port of the
paper's code ([mackelab/jaxley_experiments](https://github.com/mackelab/jaxley_experiments))
and takes the paper's data and, where available, its results from Zenodo
([14166049](https://zenodo.org/records/14166049); the Fig. 3 recordings are from
[3708064](https://zenodo.org/records/3708064)) to compare against. Paths to the data
are set at the top of each script.

All scripts run on one GPU (`--device cuda`, the default) in float64, as in the paper.

## The paper used older Jaxley versions

The paper's results were computed with Jaxley versions from mid 2024 to early 2025.
Three differences to Jaxley v0.14 (which this port follows) change the paper's models:

1. Axial conductances were not divided by the membrane capacitance (fixed in v0.3.0,
   jaxley #426). This matters where the capacitance is not 1 µF/cm²: the apical
   dendrites of the L5PC (Fig. 2) and of the RNN units (Fig. 4).
2. SWC files were read with a different reader (e.g. 339 instead of 338 branches for
   the L5PC).
3. Before 2024-08-01 (jaxley #394) there were no branchpoint nodes: a child branch was
   coupled directly to its parent, and sibling branches were not coupled. This matters
   for the RNN units of Fig. 4, whose three one-compartment branches meet at one point.

`legacy_jaxley.py` restores all three (`use_legacy_axial_conductances()`,
`read_swc_legacy()`, `use_legacy_branch_coupling()`). Which of them each experiment
needs was established by reproducing a published trace or loss. The scripts use these
settings by default (`--legacy 1`, and `--legacy-coupling` in the Fig. 4 scripts):

| Experiment | Paper's Jaxley | Evidence |
|---|---|---|
| Fig. 2a–c, 2e–i (L5PC) | 1, 2 | ground truth of Fig. 2b to 0.03 mV (all 8 spike times), of Fig. 2f to 4e-8 mV; adding 3 makes Fig. 2b 47 mV off |
| Fig. 2d (Allen cells, Jaxley v0.4.0) | 2 | |
| Fig. 3 (retina; capacitance 1) | 2 | the paper's trained model; adding 3 lowers its correlations (val 0.223 instead of 0.249; paper 0.255) |
| Fig. 4c (Lyapunov exponents, computed by 2024-07-30) | 1, 3 | one step of the map equals the paper's code on the Jaxley of 2024-07-30 to round-off (2.6e-15) |
| Fig. 4d–f (evidence integration, run of 2024-08-02) | 1 | loss of one trial printed in the paper's notebook (0.6844109395442985) to 1e-12; adding 3 makes it 4e-4 off |
| Fig. 4g (delayed match-to-sample network of the Zenodo record, trained 2024-07-19) | 1, 3 | by date; the trained network solves the task with or without 3 |
| Fig. 5 (MNIST; capacitance 1) | 2 | output traces of the paper's trained network (64 hidden and 10 output cells, 256 test images) to 5e-6 mV, identical predictions; adding 3 makes them 116 mV off |

The paper-era Jaxley itself (commit 760e770 with jaxley-mech e25512b, unchanged apart
from four API renames needed by current JAX) and the paper's own scripts were run for
the comparisons of Fig. 4c (`/nrs/scicompsoft/orhane/probes/jaxley_pytorch/paper/fig4_lyapunov/oldjax`).

Some of the paper's pickles are needed; they are read with a restricted unpickler that
only reconstructs numpy arrays and pandas DataFrames
(`/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/safe_pickle.py`).

## Results

| Figure | Script | Result (this port) | Paper |
|---|---|---|---|
| 1e speed | `bench/bench_ca1_torch_v3.py` (probe) | CA1 cell (n144, 20 ms) on an H200, compiled: simulation 0.26 s (1 sim) … 1.73 s (10,000 in parallel); gradient 1.34 s … 6.1 s | JAX (same GPU, v0.14): 0.29 s … 3.02 s; 1.64 s … 19.3 s |
| 1f large network | `fig1_large_network.py` | 2,001 cells, 719,674 compartments, 1,011,355 synapses, 3,170,377 parameters (as in the paper). A100: simulation 15.7 s, gradient 51.9 s (H100: 11.5 s, 38.7 s). Building the network takes 36–49 min | A100: 21 s, 144 s |
| 2a–c L5PC, 19 parameters | `fig2_l5pc_synthetic.py` | first two gradient-descent steps identical to the paper's for all 10 runs; 4/10 runs below loss 1 within 10 steps (best 0.33) | 5/10 (best 0.33) |
| 2d Allen cells | `fig2_allen.py`, `allen_channels.py` | 1,000 runs × 50 steps per cell in 20–56 min on an H100; the loss statistics follow the paper's logs. Weighted summary-statistics error of the selected run: 23.6, 46.6, 14.9, 19.7 (specimens 473601979, 480353286, 488683425, 485574832) | 45.0, 34.3, 15.8, 28.8; about 85 min per cell |
| 2e–i L5PC, 1,390 parameters | `fig2_l5pc_voltage_imaging.py` | observation identical (4e-8 mV); gradient descent reaches loss < 0.5 in 185 steps; HMC (200 chains) acceptance 0.59–0.81 | loss < 0.5 within 500 steps; acceptance ≈ 0.65 |
| 2j–m nonlinear neuron | `fig2_nonlinear_neuron.py` | the same 4/10 seeds learn the task (100% test accuracy); the other seeds get stuck at the same losses as in the paper | 4/10 |
| 3 retina | `fig3_retina.py` | paper's trained model: mean correlation train 0.251, val 0.249, test 0.241 (difference unexplained). Trained here (1,000 iterations, as in the paper): best validation correlation at epoch 22, train 0.260, val 0.256, test 0.244; 17–18 min per epoch on an H100 | best at epoch 22: 0.252, 0.255, 0.247; about 40 min per epoch |
| 4b–c RNN dynamics | `fig4_lyapunov.py`, `rnn_utils.py` | stable / regular / irregular activity at the three gains. Maximal Lyapunov exponent (1/ms) of seed 1 at gains 0.00125 / 0.002 / 0.00275, averaged over 24 s: 0.0134 / 0.0293 / 0.0454; the paper's code on the paper's Jaxley: 0.0149 / 0.0301 / 0.0419 (the estimates fluctuate by about ±0.004; over the paper's 6 s: 0.0109 / 0.0313 / 0.0447 and 0.0128 / 0.0354 / 0.0467). Below gain 0.00125: −0.0024 … −0.0023 | stored results: −0.0 / 0.0499 / 0.0729, which the paper's code does not reproduce either; −0.0024 … −0.0023 below gain 0.00125 |
| 4d–f evidence integration | `fig4_evidence_integration.py` | paper's trained network: 100% accuracy (1,000 trials), psychometric curve as in the paper. Trained here (2,000 steps of batch 4, as in the paper): 100% accuracy, with the input scaled as in the paper's training (4) or its notebook (1) | 99.9% |
| 4g delayed match-to-sample | `fig4_dms.py` | paper's trained network: 100% accuracy (512 trials, delays 450–550 ms); training *running* (reached the final delay at step 126) | all four patterns correct |
| 5 MNIST | `fig5_mnist.py` | paper's trained network: 94.23% test accuracy, output traces as in the paper (5e-6 mV); parameter resets and adversarial attacks as in the paper; one training step (batch 16) 2.7 s on an H100 | 94.2%; 25 s per step on a V100 |

Not reproduced: comparisons with the NEURON simulator (Fig. 1d; no NEURON build for
Python 3.14) and the genetic-algorithm baselines (Figs. 2c, 2i, 2m; BluePyOpt/efel have
no Python 3.14 builds) — the plots show the paper's genetic-algorithm curves instead.
