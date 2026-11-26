"""A comparacao de especificacoes do ARIMA nao pode usar meses do holdout."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

pd = pytest.importorskip("pandas")
ar = pytest.importorskip("arima_residuals")


def test_windows_end_before_holdout():
    series = ar.load_series(ar.DATA_FILE)
    train, valid = ar.split_windows(series)
    first_target = series.index[-ar.HOLDOUT_MONTHS]
    assert train.index.max() < valid.index.min()
    assert valid.index.max() < first_target
    assert len(valid) == ar.VALIDATION_MONTHS


def test_partial_first_month_excluded():
    series = ar.load_series(ar.DATA_FILE)
    assert series.index.min() == pd.Timestamp("2006-04-01")


def test_partial_last_month_excluded():
    # exportacao CEPEA em 07/04/2026: abril/2026 e' parcial e fica fora
    series = ar.load_series(ar.DATA_FILE)
    assert series.index.max() == pd.Timestamp("2026-03-01")


def test_first_holdout_target_matches_holdout_protocol():
    data = pd.read_csv(ROOT / "data" / "processed" / "features_monthly_modeling.csv",
                       parse_dates=["date"])
    last_origins = data["date"].sort_values().iloc[-12:]
    first_target = last_origins.iloc[0] + pd.DateOffset(months=1)
    series = ar.load_series(ar.DATA_FILE)
    assert first_target == series.index[-ar.HOLDOUT_MONTHS]
    assert data["target_month"].max() == series.index.max().strftime("%Y-%m-%d")
