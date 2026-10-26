# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Where a run's network starts when it is a teacher's (`[init]`, see
`config.InitConfig`): the teacher's connectivity, weights, inputs and readout, with
gains and biases as `start` says:

  shuffled  the teacher's, permuted within the excitatory and within the inhibitory
            neurons, gains halved (the paper's students, `nn_fig1_trainStudent.py`);
  random    new draws from the teacher's initial distribution
            (`rate_network.random_gains_biases`);
  uniform   gains 1, biases 0;
  teacher   the teacher's own (a control: how far training alone moves a network away
            from the teacher);
  student   those of a trained student (its run directory).
"""

import math
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from jaxley_pytorch.training.rate_network import RateNetwork, random_gains_biases

STARTS = ("shuffled", "random", "uniform", "teacher", "student")


def make_student(
    teacher: RateNetwork,
    conn: Dict,
    start: str,
    seed: int,
    student_dir: Optional[str] = None,
    gain_scale: float = 1.0,
) -> RateNetwork:
    """A network with the teacher's weights, inputs and readout and the gains and biases
    of `start` (drawn with `seed`), the gains then multiplied by `gain_scale`."""
    g = teacher.g.detach().cpu().double().numpy().copy()
    b = teacher.b.detach().cpu().double().numpy().copy()
    rng = np.random.default_rng(seed)
    if start == "student":
        path = os.path.join(student_dir, "student.pt")
        st = torch.load(path, map_location="cpu", weights_only=True)
        g, b = st["g"].double().numpy(), st["b"].double().numpy()
    elif start == "shuffled":
        for part in (slice(0, conn["n_exc"]), slice(conn["n_exc"], None)):
            g[part] = rng.permutation(g[part])
            b[part] = rng.permutation(b[part])
        g = 0.5 * g
    elif start == "random":
        g, b = random_gains_biases(len(g), rng)
    elif start == "uniform":
        g, b = np.ones(len(g)), np.zeros(len(b))
    elif start != "teacher":
        raise ValueError(f"[init] start must be one of {STARTS}, got {start!r}")
    io = {k: v.detach() for k, v in teacher.io.items()}
    return RateNetwork(teacher.sim, teacher.W.detach(), gain_scale * g, b, io)


def stable_student(
    teacher: RateNetwork,
    conn: Dict,
    start: str,
    seed: int,
    start_loss: Callable[[RateNetwork], float],
    silent_loss: float,
) -> Tuple[RateNetwork, Dict]:
    """A `random` or `uniform` start, drawn again (seed + 1,000 x attempt; uniform: the
    gains x 0.9 per attempt) while its task loss `start_loss(net)` is not finite or
    above 10 times `silent_loss`, the loss of a silent readout: its activity runs away.
    Returns the network and {"redraws", "start_loss"}."""
    attempt = 0
    net = make_student(teacher, conn, start, seed)
    while True:
        loss = start_loss(net)
        if math.isfinite(loss) and loss < 10 * silent_loss:
            return net, {"redraws": attempt, "start_loss": loss}
        attempt += 1
        if attempt > 50:
            raise RuntimeError(
                f"no stable start in 50 attempts (last task loss {loss:.3g}, limit "
                f"{10 * silent_loss:.3g})"
            )
        scale = 0.9**attempt if start == "uniform" else 1.0
        net = make_student(
            teacher, conn, start, seed + 1000 * attempt, gain_scale=scale
        )
