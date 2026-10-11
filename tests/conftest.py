# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Fixtures (ported from Jaxley's `tests/conftest.py`).

All tests run in float64, the counterpart of Jaxley's `jax_enable_x64`.
"""

import os
import warnings
from copy import deepcopy
from typing import Optional

import pytest
import torch

torch.set_default_dtype(torch.float64)

import jaxley_pytorch as jx
from jaxley_pytorch.channels import AdEx
from jaxley_pytorch.synapses import IonotropicSynapse


def _binary_tree_parents(nbranches: int) -> list:
    parents = [-1]
    depth = 0
    while nbranches > len(parents):
        parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
        depth += 1
    return parents[:nbranches]


@pytest.fixture(scope="session")
def SimpleComp():
    """Fixture for creating or retrieving an already created compartment."""
    comps = {}

    def get_or_build_comp(
        copy: bool = True, force_init: bool = False
    ) -> jx.Compartment:
        if "comp" not in comps or force_init:
            comps["comp"] = jx.Compartment()
        return deepcopy(comps["comp"]) if copy and not force_init else comps["comp"]

    yield get_or_build_comp


@pytest.fixture(scope="session")
def SimpleBranch(SimpleComp):
    """Fixture for creating or retrieving an already created branch."""
    branches = {}

    def get_or_build_branch(
        ncomp: int, copy: bool = True, force_init: bool = False
    ) -> jx.Branch:
        if ncomp not in branches or force_init:
            comp = SimpleComp(force_init=force_init)
            branches[ncomp] = jx.Branch([comp] * ncomp)
        return deepcopy(branches[ncomp]) if copy and not force_init else branches[ncomp]

    yield get_or_build_branch


@pytest.fixture(scope="session")
def SimpleCell(SimpleBranch):
    """Fixture for creating or retrieving an already created cell.

    The branch structure is assumed as [-1, 0, 0, 1, 1, 2, 2, ...].
    """
    cells = {}

    def get_or_build_cell(
        nbranches: int, ncomp: int, copy: bool = True, force_init: bool = False
    ) -> jx.Cell:
        key = (nbranches, ncomp)
        if key not in cells or force_init:
            parents = _binary_tree_parents(nbranches)
            branch = SimpleBranch(ncomp=ncomp, force_init=force_init)
            cells[key] = jx.Cell([branch] * nbranches, parents)
        return deepcopy(cells[key]) if copy and not force_init else cells[key]

    yield get_or_build_cell


@pytest.fixture(scope="session")
def SimpleNet(SimpleCell):
    """Fixture for creating or retrieving an already created network."""
    nets = {}

    def get_or_build_net(
        ncells: int,
        nbranches: int,
        ncomp: int,
        connect: bool = False,
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Network:
        key = (ncells, nbranches, ncomp, connect)
        if key not in nets or force_init:
            net = jx.Network(
                [SimpleCell(nbranches=nbranches, ncomp=ncomp, force_init=force_init)]
                * ncells
            )
            if connect:
                jx.connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
            nets[key] = net
        return deepcopy(nets[key]) if copy and not force_init else nets[key]

    yield get_or_build_net


@pytest.fixture(scope="session")
def SpikeNet(SimpleComp):
    """Fixture for creating a network with AdEx spiking neurons."""
    nets = {}

    def get_or_build_net(
        ncells: int,
        nbranches: int,
        ncomp: int,
        connect: bool = False,
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Network:
        key = (ncells, nbranches, ncomp, connect)
        if key not in nets or force_init:
            cells = []
            for _ in range(ncells):
                parents = _binary_tree_parents(nbranches)
                comp = SimpleComp(force_init=force_init)
                branch = jx.Branch([comp] * ncomp)
                cell = jx.Cell([branch] * nbranches, parents)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Ignore surrogate warning
                    cell.insert(AdEx())
                cell.set("capacitance", 200.0)
                cells.append(cell)

            net = jx.Network(cells)
            if connect and ncells >= 2:
                jx.connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
            nets[key] = net
        return deepcopy(nets[key]) if copy and not force_init else nets[key]

    yield get_or_build_net


@pytest.fixture(scope="session")
def SimpleMorphCell():
    """Fixture for creating or retrieving an already created morpholgy."""
    cells = {}

    def get_or_build_cell(
        fname: Optional[str] = None,
        ncomp: int = 1,
        max_branch_len: float = 2_000.0,
        ignore_swc_tracing_interruptions: bool = True,
        swc_backend: str = "graph",
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Cell:
        dirname = os.path.dirname(__file__)
        default_fname = os.path.join(dirname, "swc_files", "morph_ca1_n120.swc")
        fname = default_fname if fname is None else fname
        key = (fname, ncomp, max_branch_len, ignore_swc_tracing_interruptions)
        if key not in cells or force_init:
            cells[key] = jx.read_swc(
                fname,
                ncomp=ncomp,
                max_branch_len=max_branch_len,
                assign_groups=True,
                backend=swc_backend,
                ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
            )
        return deepcopy(cells[key]) if copy and not force_init else cells[key]

    yield get_or_build_cell


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
