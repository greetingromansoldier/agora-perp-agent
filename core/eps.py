"""float comparison helpers with a single project-wide epsilon."""

from __future__ import annotations

CROSS_EPS: float = 1e-12


def gt_eps(a: float, b: float) -> bool:
    return a > b + CROSS_EPS


def lt_eps(a: float, b: float) -> bool:
    return a + CROSS_EPS < b


def le_eps(a: float, b: float) -> bool:
    return a <= b + CROSS_EPS


def ge_eps(a: float, b: float) -> bool:
    return a + CROSS_EPS >= b
