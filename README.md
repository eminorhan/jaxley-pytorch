# jaxley-pytorch

A PyTorch port of [Jaxley](https://github.com/jaxleyverse/jaxley) ([Deistler et al.,
*Nature Methods* 2025](https://www.nature.com/articles/s41592-025-02895-w)), 
a differentiable simulator for biophysically detailed neuron models. 
The port follows Jaxley `v0.14.0` (commit `61675ec`) and keeps its API, so models are built exactly as in Jaxley:

```python
import torch
import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH

cell = jx.Cell()
cell.insert(HH())
cell.stimulate(jx.step_current(i_delay=1.0, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=10.0))
cell.record("v")
v = jx.integrate(cell)  # torch.Tensor of shape (num_recordings, num_timesteps)
```

The ion channels of [jaxley-mech](https://github.com/jaxleyverse/jaxley-mech) are implemented in `jaxley_pytorch.mech` (`l5pc`, `fohlmeister97`).

## Differences from Jaxley

* **Gradients** use PyTorch autograd. Mark parameters as trainable as in Jaxley and pass
  leaf tensors to `jx.integrate`:

  ```python
  cell.make_trainable("HH_gNa")
  params = [{k: v.clone().requires_grad_(True) for k, v in p.items()}
            for p in cell.get_parameters()]
  loss = jx.integrate(cell, params=params).mean()
  loss.backward()
  ```

* **Batching replaces `jax.vmap`.** Every input may carry leading batch dimensions, which
  are broadcast against each other: trainable parameters `(..., num_params)`, `data_set`
  values `(..., num_compartments)` (e.g. `(batch, 1)` for one value per simulation),
  stimuli and clamps `(..., num_inputs, time)`, and states `(..., num_compartments)`.
  The recordings then have shape `(..., num_recordings, time)`.
* **Device and dtype** follow the inputs: the simulation runs on the device and with the
  dtype of the trainable parameters, `param_state`, stimuli or initial states (whichever
  is given first), otherwise on the default torch device and dtype. Use
  `torch.set_default_dtype(torch.float64)` for the equivalent of `jax_enable_x64`.
* **No `jit`.** Simulations run eagerly by default. `jx.integrate(..., compile=True)`
  compiles the integration step with `torch.compile`.
* **GPU tree solver.** On CUDA, the voltage equations are solved with a fused Triton
  kernel (one kernel launch per time step instead of one per tree level). It is
  differentiable (twice) and deterministic. There are two variants, chosen per
  morphology from an estimate of their work: one holds a block of small trees in
  registers, the other touches only the compartments of each tree level (much faster
  for deep trees such as reconstructed morphologies). `JAXLEY_PYTORCH_TREE_KERNEL=
  registers|levels` forces one.
* **CUDA graphs.** `jx.integrate` makes no host-device copies once a module has been
  simulated, so a whole training iteration (simulation, loss, backward, optimizer step;
  use `torch.optim` optimizers with `capturable=True`) can be captured with
  `torch.cuda.graph` and replayed. For small models, whose cost is dominated by launching
  kernels, this is 10-50x faster (see `examples/paper/fig2_nonlinear_neuron.py`).
* **Forward-mode derivatives.** `torch.func.jvp` and `torch.autograd.forward_ad` work
  through `jx.integrate` and through the step function of `build_init_and_step_fn` (e.g.
  for Lyapunov exponents, `examples/paper/fig4_lyapunov.py`).
* **float32.** Gating rates of the form `x / (exp(x) - 1)` (`HH`, the Pospischil
  channels, jaxley-mech's `efun`) are evaluated with `x_over_expm1` (finite at their
  removable singularity, which float32 voltages hit now and then), and the time constant
  of `IonotropicSynapse` without the cancellation in `1 - s_inf` (which made its gradient
  NaN once a steep release saturates). The float64 results are unchanged (to
  round-off); `tests/test_float32_safety.py`.
* **Synapses ordered by postsynaptic compartment.** For a synapse type whose synapses
  are ordered by postsynaptic compartment, with the same number of synapses per
  compartment (e.g. a network with a fixed in-degree, built one postsynaptic cell after
  the other), the currents of every compartment's synapses are summed
  (`Network._step_regular_synapses`) instead of added up with `index_add`, which
  `torch.compile` turns into one atomic add per synapse. On CUDA, the two synapse
  models added by the port, `KineticSynapse` (first-order kinetics) and
  `SigmoidRateSynapse` (a current-based rate synapse), are then simulated with fused
  Triton kernels (`jaxley_pytorch/kernels/synapse_step.py`; one kernel per pass, no
  atomic adds). The results are the same up to round-off
  (`tests/test_regular_synapses.py`, `tests/test_synapse_kernels.py`); together, the
  networks of `examples/language_modeling` train 1.7-4.5x faster with 2.6-8x less
  memory. For such synapses, `jx.integrate` computes the current through every
  synapse (`i_<synapse>`) only if it is recorded, so the states returned with
  `return_states=True` do not contain it otherwise.
* **Renamed:** `module.to_jax()` is `module.to_torch()`, `.jaxnodes` / `.jaxedges` are
  `.torchnodes` / `.torchedges`. `TypeOptimizer` wraps `torch.optim` optimizers
  (`step()` / `zero_grad()`) instead of `optax`.
* **Default voltage solver:** Jaxley solves the voltage equations sequentially on CPU
  (one compartment per step of a compiled loop) and level by level on GPU. Eager
  PyTorch pays per operation, so the level-parallel solver is the default on all
  devices (6x faster for an L5PC on CPU); results differ from Jaxley's CPU results
  only by round-off. `voltage_solver="jaxley.dhs.cpu"` (after
  `_init_solver_jaxley_dhs_solve(allowed_nodes_per_level=1)`) gives the sequential
  solver.
* `voltage_solver="jax.sparse"` is a dense direct solve (`torch.linalg.solve`), meant for
  testing on small morphologies. `voltage_solver="jaxley.stone"` (unbranched cables) uses
  the DHS solver.

Model construction (the `.nodes` / `.edges` tables, views, SWC import, morphology
editing) follows Jaxley's pandas/numpy/networkx code, nearly unchanged.

## The paper's examples

`examples/paper/` reproduces the experiments of the Jaxley paper (Figs. 1–5) with this
port, see `examples/paper/README.md`. The paper's results were computed with older
Jaxley versions, whose behaviour differs in three places (axial conductances, the SWC
reader and the coupling of branches); `examples/paper/legacy_jaxley.py` restores it so
that the paper's numbers are reproduced.

## Another fun example: language models of biophysical neurons

`examples/language_modeling/` is a prototype that trains recurrent networks of rate,
Hodgkin-Huxley, Pospischil and three-compartment L5PC neurons (sparse, Dale's law) as
character-level language models on Andrej Karpathy's cleaned-up [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) dataset, with throughput measurements up to
65,536 neurons, see `examples/language_modeling/README.md` for more details.

## Tests

The upstream test suite is ported to `tests/` (including the golden values of
`tests/jaxley_identical`, compared in float64 at a tolerance of 1e-8):

```bash
python -m pytest tests            # add --runslow for the slow tests
```

Tests that need the NEURON simulator are skipped if it is not installed.
`tests/test_kernels.py` needs a GPU.

## License

Apache License 2.0 (see `LICENSE`), as Jaxley. The channels in `jaxley_pytorch/mech` are
ported from `jaxley-mech`, which is MIT licensed (`jaxley_pytorch/mech/LICENSE`).

## Acknowledgments

The code in this repo is almost entirely written by Claude Opus 5.5 Max. 
If you find this repo useful for your own work, please cite the original [Jaxley paper](https://www.nature.com/articles/s41592-025-02895-w):

```
@article{deistler2025,
  title={Jaxley: Differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics},
  author={Deistler, Michael and Kadhim, Kyra L and Pals, Matthijs and Beck, Jonas and Huang, Ziwei and Gloeckler, Manuel and Lappalainen, Janne K and Schr{\"o}der, Cornelius and Berens, Philipp and Gon{\c{c}}alves, Pedro J and others},
  journal={Nature Methods},
  volume={22},
  number={12},
  pages={2649--2657},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```

