# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""`jaxley_pytorch.training.spiking_network`: the connectivity, the single-neuron
parameters, the simulation, spike counts and the students' starts."""

import numpy as np
import pytest
import torch

from jaxley_pytorch.training import init
from jaxley_pytorch.training import spiking_network as sn


def model(name="pospischil", **kwargs):
    args = dict(
        neurons=10,
        exc_frac=0.8,
        fan_in=5,
        dt=0.025,
        input_scale=2.0,
        w_exc=0.01,
        w_inh=0.05,
        spread=0.25,
    )
    args.update(kwargs)
    return sn.SPIKING_MODELS[name](**args)


def init_io(n, rng, alpha):
    return {
        "embed": rng.standard_normal((3, 2)),
        "embed_proj": rng.standard_normal((n, 2)),
    }


def network(name="pospischil", seed=0, **kwargs):
    m = model(name, **kwargs)
    net, conn = sn.new_network(
        m, init_io, seed, torch.device("cpu"), torch.float64, False
    )
    return net, conn, m


def test_connectivity():
    """Every neuron receives k_exc excitatory and fan_in - k_exc inhibitory synapses,
    none from itself, listed by postsynaptic neuron."""
    m = model(neurons=20, fan_in=6)
    rng = np.random.default_rng(0)
    pre, post = sn.sample_connectivity(20, m.n_exc, 6, m.k_exc, rng)
    assert np.all(np.diff(post) >= 0) and np.all(pre != post)
    for j in range(20):
        mine = pre[post == j]
        assert len(mine) == 6 and len(set(mine)) == 6
        assert (mine < m.n_exc).sum() == m.k_exc


def test_settings():
    """Too few neurons of a type for the fan-in is an error."""
    with pytest.raises(ValueError, match="needs more neurons of each type"):
        model(neurons=10, fan_in=9)


@pytest.mark.parametrize("name", ["pospischil", "hh"])
def test_parameters(name):
    """Log conductances around the defaults of every type (Pospischil's M-type K of the
    excitatory neurons only) and biases 0."""
    net, conn, m = network(name)
    expected = {
        "pospischil": ("gNa", "gK", "gKm", "gLeak", "bias"),
        "hh": ("gNa", "gK", "gLeak", "bias"),
    }
    assert net.neuron_param_names == expected[name]
    assert tuple(p for p in m.PARAMETERS if p != "W") == expected[name]
    defaults = sn.default_conductances(m)
    for cond, values in defaults.items():
        log_g = net.log_g[cond].detach().numpy()
        assert log_g.shape == values.shape
        assert np.all(np.abs(log_g - np.log(values)) < 6 * m.spread)
    if name == "pospischil":
        assert len(net.log_g["gKm"]) == m.n_exc
        assert defaults["gK"][0] != defaults["gK"][-1]  # regular vs fast spiking
    assert torch.all(net.bias == 0)


def test_neuron_parameters_act_on_their_neuron():
    """A neuron's bias and conductances change its own voltage, and (below the
    synapses' release threshold) not the other neurons'."""
    net, conn, m = network()
    current = torch.zeros(2, 10, 40, dtype=torch.float64)
    v, after = net.sim.segment(net.sim.rest_states(2), current, net.sim_params())
    assert v.shape == (2, 10, 40) and set(after) == set(net.sim.rest_states(2))
    with torch.no_grad():
        net.bias[3] = 5.0
        net.log_g["gLeak"][7] += 2.0
    v2, _ = net.sim.segment(net.sim.rest_states(2), current, net.sim_params())
    assert torch.all(v2[:, 3, -1] > v[:, 3, -1] + 0.1)
    assert torch.all((v2[:, 7, -1] - v[:, 7, -1]).abs() > 0.01)
    others = [i for i in range(10) if i not in (3, 7)]
    assert torch.allclose(v2[:, others], v[:, others])


def test_dale():
    """Excitatory synapses reverse at 0 mV and inhibitory ones at -80 mV, and their
    conductances are never negative."""
    net, conn, m = network()
    name = net.sim.name
    e_syn = net.sim.net.edges[f"{name}_e_syn"].to_numpy()
    is_exc = np.asarray(conn["pre"]) < m.n_exc
    assert np.all(e_syn[is_exc] == 0.0) and np.all(e_syn[~is_exc] == -80.0)
    jaxley, _ = net.sim.params(-50 * torch.ones_like(net.W), dict(net.log_g), net.bias)
    assert torch.all(jaxley[0][f"{name}_gS"] >= 0)


def test_release_width():
    """The synapses' release slope is the model's release_width (2 mV by default), and
    the settings record it."""
    net, _, m = network()
    name = net.sim.name
    assert np.all(net.sim.net.edges[f"{name}_delta"].to_numpy() == 2.0)
    assert m.settings()["release_width"] == 2.0
    wide, _, m5 = network(release_width=5.0)
    assert np.all(wide.sim.net.edges[f"{name}_delta"].to_numpy() == 5.0)
    assert m5.settings()["release_width"] == 5.0
    with pytest.raises(ValueError, match="release_width must be positive"):
        model(release_width=0.0)


@pytest.mark.parametrize("name", ["pospischil", "hh"])
def test_na_activation_gradient(name):
    """Below 1, the sodium activation gate reads the voltage with that derivative: its
    update keeps its value, its derivative is scaled and the other gates' are not; the
    simulation and its spikes are the same and the gradient differs. 1 (the default)
    keeps the exact channels; the settings record it."""
    exact_cls = sn.Na if name == "pospischil" else sn.HH
    exact, soft = exact_cls(), sn.SURROGATES[exact_cls](0.25)
    v = torch.linspace(-80.0, 30.0, 12, dtype=torch.float64, requires_grad=True)
    params = {"vt": torch.tensor(-60.0, dtype=torch.float64)}
    states = exact.init_state({}, v.detach(), params, 0.025)

    def update(channel):
        out = channel.update_states(states, 0.025, v, params)
        return {
            k: (x, torch.autograd.grad(x.sum(), v, retain_graph=True)[0])
            for k, x in out.items()
        }

    soft_out = update(soft)
    for key, (value, grad) in update(exact).items():
        assert torch.equal(soft_out[key][0], value)
        scale = 0.25 if key.endswith("_m") else 1.0
        assert torch.allclose(soft_out[key][1], scale * grad)

    nets = {g: network(name, na_activation_gradient=g) for g in (1.0, 0.25)}
    current = torch.full((1, 10, 800), 5.0, dtype=torch.float64)
    out = {}
    for g, (net, _, m) in nets.items():
        assert m.settings()["na_activation_gradient"] == g
        kinds = {type(c) for c in net.sim.net.channels}
        assert (exact_cls in kinds) == (g == 1.0)
        net.set_trainable(["W", "gNa"])
        rest = net.sim.rest_states(1)
        volts, _ = net.sim.segment(rest, current, net.sim_params())
        net.features(volts).mean().backward()
        out[g] = (volts.detach(), net.spike_counts(rest["v"], volts).sum())
        out[g] += (net.W.grad, net.log_g["gNa"].grad)
    assert out[1.0][1] > 0 and out[0.25][1] == out[1.0][1]
    assert torch.equal(out[0.25][0], out[1.0][0])
    for k in (2, 3):
        assert not torch.allclose(out[0.25][k], out[1.0][k])
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="na_activation_gradient must be from 0"):
            model(na_activation_gradient=bad)


def test_spike_counts_and_features():
    net, _, _ = network()
    v = torch.tensor([[-70.0, 10.0, 20.0, -50.0, 5.0], [5.0, 6.0, -10.0, -20.0, -30.0]])
    before = torch.tensor([-70.0, 3.0])
    assert net.spike_counts(before, v).tolist() == [2.0, 0.0]
    assert torch.allclose(
        net.features(torch.tensor([-60.0, -40.0])), torch.tensor([0.0, 1.0])
    )


def test_student_starts():
    """shuffled: every single-neuron parameter permuted within the excitatory and within
    the inhibitory neurons; teacher: the teacher's own; uniform is not available."""
    teacher, conn, m = network(neurons=20, fan_in=6)
    with torch.no_grad():
        teacher.bias.copy_(torch.arange(20.0))
    student = init.make_student(teacher, conn, "shuffled", seed=1)
    n_exc = m.n_exc
    for name, t in teacher.neuron_params().items():
        s, t = student.neuron_params()[name].detach().numpy(), t.detach().numpy()
        parts = [slice(0, n_exc), slice(n_exc, None)] if len(t) == 20 else [slice(None)]
        for part in parts:
            assert sorted(s[part]) == sorted(t[part])
        assert not np.array_equal(s, t)
    assert torch.equal(student.W, teacher.W) and student.sim is teacher.sim
    same = init.make_student(teacher, conn, "teacher", seed=1)
    for name, t in teacher.neuron_params().items():
        assert torch.equal(same.neuron_params()[name], t)
    with pytest.raises(ValueError, match="not available for spiking networks"):
        init.make_student(teacher, conn, "uniform", seed=1)


def test_random_start():
    """random: new draws from the teacher's initial distribution with the student's seed
    (log conductances around the defaults of every type, biases 0), whatever the
    teacher's trained values; the teacher's weights, inputs and readout."""
    teacher, conn, m = network(neurons=20, fan_in=6)
    with torch.no_grad():
        teacher.bias.copy_(torch.arange(20.0))
        for v in teacher.log_g.values():
            v.add_(5.0)
    student = init.make_student(teacher, conn, "random", seed=1)
    again = init.make_student(teacher, conn, "random", seed=1)
    other = init.make_student(teacher, conn, "random", seed=2)
    for name, values in sn.default_conductances(m).items():
        log_g = student.log_g[name].detach().numpy()
        assert log_g.shape == values.shape
        assert np.all(np.abs(log_g - np.log(values)) < 6 * m.spread)
        assert torch.equal(student.log_g[name], again.log_g[name])
        assert not torch.equal(student.log_g[name], other.log_g[name])
    assert torch.all(student.bias == 0)
    assert torch.equal(student.W, teacher.W) and student.sim is teacher.sim
    for k, v in teacher.io.items():
        assert torch.equal(student.io[k], v)
