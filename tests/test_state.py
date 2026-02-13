"""Tests for state generators and utilities."""
import math
import numpy as np
import pytest

from qsp.state import (
    cardinality,
    dense_uniform,
    dicke_state,
    get_supports,
    ground_state,
    normalize,
    random_dense_state,
    random_sparse_state,
    sparse_random,
    sparse_uniform,
    to_statevector,
    w_state,
)


def _norm(state):
    return math.sqrt(sum(v * v for v in state.values()))


class TestGenerators:
    def test_ground_state(self):
        s, n = ground_state(3)
        assert n == 3
        assert s == {0: 1.0}

    def test_w_state(self):
        s, n = w_state(3)
        assert n == 3
        assert cardinality(s) == 3
        assert abs(_norm(s) - 1.0) < 1e-10

    def test_dicke_state_k2(self):
        s, n = dicke_state(4, 2)
        assert n == 4
        assert cardinality(s) == 6
        assert abs(_norm(s) - 1.0) < 1e-10

    def test_dense_uniform(self):
        s, n = dense_uniform(4)
        assert n == 4
        assert cardinality(s) == 8
        assert abs(_norm(s) - 1.0) < 1e-10

    def test_sparse_uniform(self):
        s, n = sparse_uniform(4, seed=1)
        assert n == 4
        assert cardinality(s) == 4
        assert abs(_norm(s) - 1.0) < 1e-10

    def test_sparse_random(self):
        s, n = sparse_random(4, seed=1)
        assert n == 4
        assert cardinality(s) == 4
        assert abs(_norm(s) - 1.0) < 1e-10

    def test_random_dense(self):
        s, n = random_dense_state(4, seed=1)
        assert n == 4
        assert cardinality(s) == 8
        assert abs(_norm(s) - 1.0) < 1e-10

    def test_random_sparse(self):
        s, n = random_sparse_state(4, seed=1)
        assert n == 4
        assert cardinality(s) == 4
        assert abs(_norm(s) - 1.0) < 1e-10


class TestUtilities:
    def test_normalize(self):
        s = {0: 3.0, 1: 4.0}
        ns = normalize(s)
        assert abs(_norm(ns) - 1.0) < 1e-10

    def test_get_supports(self):
        s = {0b001: 0.5, 0b101: 0.5}
        supports = get_supports(s, 3)
        assert 0 in supports
        assert 2 in supports

    def test_to_statevector(self):
        s, n = w_state(3)
        sv = to_statevector(s, n)
        assert sv.shape == (8,)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10
