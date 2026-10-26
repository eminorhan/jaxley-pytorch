# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Training networks of jaxley_pytorch neurons on tasks, one TOML config per arm of an
experiment (not part of Jaxley):

    python -m jaxley_pytorch.training --config <arm>.toml --output-root <dir> [--seed k]

  config.py        the config file
  run.py           the entrypoint and the run directory
  engine.py        the training loop (Adam, CUDA graphs, checkpoints; L-BFGS)
  rate_network.py  Beiran & Litwin-Kumar's rate network ([model] name = "blk_rate")
  tasks.py         the tasks ([task] name = "cycling" or "text")
  init.py          networks that start from a trained one, the teacher ([init])
  metrics.py       teacher-student comparisons
  report.py        what the experiments' summary scripts share

The experiments are in examples/activitome-connectome/.
"""
