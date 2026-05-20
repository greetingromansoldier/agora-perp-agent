from __future__ import annotations

import pytest

from core.eps import CROSS_EPS, ge_eps, gt_eps, le_eps, lt_eps


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1.0 + 2e-12, 1.0, True),
        (1.0 + 1e-13, 1.0, False),
        (1.0, 1.0, False),
        (1.0, 2.0, False),
    ],
)
def test_gt_eps(a, b, expected):
    assert gt_eps(a, b) is expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1.0, 1.0 + 2e-12, True),
        (1.0, 1.0 + 1e-13, False),
        (2.0, 1.0, False),
    ],
)
def test_lt_eps(a, b, expected):
    assert lt_eps(a, b) is expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1.0, 1.0, True),
        (1.0 - 1e-13, 1.0, True),
        (1.0 + 2e-12, 1.0, False),
    ],
)
def test_le_eps(a, b, expected):
    assert le_eps(a, b) is expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1.0, 1.0, True),
        (1.0 + 1e-13, 1.0, True),
        (1.0, 1.0 + 2e-12, False),
    ],
)
def test_ge_eps(a, b, expected):
    assert ge_eps(a, b) is expected


def test_cross_eps_value():
    assert CROSS_EPS == 1e-12
