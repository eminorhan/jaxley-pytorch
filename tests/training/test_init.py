# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""`jaxley_pytorch.training.init`: the gains and biases a student starts from."""

import numpy as np
import pytest
import torch

from jaxley_pytorch.training import init
from tests.training.test_rate_network import network


def test_starts():
    teacher, conn = network(30)
    g, b = teacher.g.detach().numpy(), teacher.b.detach().numpy()
    exc, inh = slice(0, conn["n_exc"]), slice(conn["n_exc"], None)

    shuffled = init.make_student(teacher, conn, "shuffled", seed=1)
    gs, bs = shuffled.g.detach().numpy(), shuffled.b.detach().numpy()
    for part in (exc, inh):  # permuted within E and within I, gains halved
        assert np.sort(gs[part]) == pytest.approx(np.sort(0.5 * g[part]))
        assert np.sort(bs[part]) == pytest.approx(np.sort(b[part]))
    assert not np.allclose(bs, b)
    assert shuffled.sim is teacher.sim and torch.equal(shuffled.W, teacher.W)

    random = init.make_student(teacher, conn, "random", seed=1)
    assert np.all(random.g.detach().numpy() > 0)
    assert np.all(np.abs(random.b.detach().numpy()) <= 0.1)

    uniform = init.make_student(teacher, conn, "uniform", seed=1)
    assert np.all(uniform.g.detach().numpy() == 1)
    assert np.all(uniform.b.detach().numpy() == 0)

    own = init.make_student(teacher, conn, "teacher", seed=1)
    assert torch.equal(own.g, teacher.g) and torch.equal(own.b, teacher.b)
    assert not any(p.requires_grad for _, p in own.named_weights())


def test_stable_student_redraws():
    """A start whose loss runs away is drawn again with another seed (uniform: smaller
    gains)."""
    teacher, conn = network(30)
    losses = iter([float("inf"), 50.0, 5.0])
    net, info = init.stable_student(
        teacher, conn, "uniform", 3, lambda _: next(losses), silent_loss=1.0
    )
    assert info == {"redraws": 2, "start_loss": 5.0}
    assert np.allclose(net.g.detach().numpy(), 0.9**2)
