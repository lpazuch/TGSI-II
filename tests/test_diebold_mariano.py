"""Testes do teste de Diebold-Mariano (experiments/diebold_mariano.py)."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

np = pytest.importorskip("numpy")
dmmod = pytest.importorskip("diebold_mariano")


def test_antisymmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50) * 1.3
    r_ab = dmmod.diebold_mariano(a, b)
    r_ba = dmmod.diebold_mariano(b, a)
    assert r_ab["dm_hln"] == pytest.approx(-r_ba["dm_hln"])
    assert r_ab["p_value"] == pytest.approx(r_ba["p_value"])


def test_identical_errors_give_nan():
    a = np.array([1.0, -2.0, 0.5, 3.0])
    r = dmmod.diebold_mariano(a, a)
    assert math.isnan(r["dm_hln"]) and math.isnan(r["p_value"])


def test_manual_value_h1():
    a = np.array([1.0, 2.0, 1.5, 3.0, 0.5])
    b = np.array([2.0, 2.5, 1.0, 3.5, 1.5])
    d = np.abs(a) - np.abs(b)
    n = len(d)
    var = np.mean((d - d.mean()) ** 2)
    dm = d.mean() / math.sqrt(var / n)
    hln = dm * math.sqrt((n - 1) / n)
    r = dmmod.diebold_mariano(a, b)
    assert r["dm_hln"] == pytest.approx(hln)
    assert r["dm_hln"] < 0  # a tem erros absolutos menores


def test_clearly_better_model_is_significant():
    rng = np.random.default_rng(1)
    good = rng.normal(scale=1.0, size=200)
    bad = rng.normal(scale=3.0, size=200)
    r = dmmod.diebold_mariano(good, bad)
    assert r["dm_hln"] < 0 and r["p_value"] < 0.01


def test_newey_west_lag():
    assert dmmod.newey_west_lag(12) == 2
    assert dmmod.newey_west_lag(216) == 4
