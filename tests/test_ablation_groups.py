"""Testes da ablacao (experiments/ablation.py):
- particao das 54 features em grupos tematicos e em variaveis de origem;
- regra de selecao (media - 2*desvio > 0);
- configuracoes 'relevantes' e 'sem_<familia>'."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

ablation = pytest.importorskip("ablation")
pd = pytest.importorskip("pandas")

BASES = ["usd_brl", "brent_usd_bbl", "precipitation_mm", "temperature_mean_c",
         "relative_humidity_pct", "wind_speed_ms", "ndvi", "soil_moisture_m3m3"]
FEATURES = (
    ["month", "quarter", "soy_planting_window", "soy_harvest_window",
     "temperature_max_c", "temperature_min_c", "soy_price_brl_bag", "soy_price_usd_bag"]
    + BASES
    + [f"{b}_{s}" for b in BASES for s in ("lag_1", "lag_2", "lag_3", "rolling_mean_3")]
    + [f"{b}_month_anomaly" for b in ("usd_brl", "brent_usd_bbl", "precipitation_mm",
                                       "temperature_mean_c", "ndvi", "soil_moisture_m3m3")]
)


def test_feature_set_has_54_columns():
    assert len(FEATURES) == len(set(FEATURES)) == 54


def test_group_sizes():
    sizes = {}
    for f in FEATURES:
        g = ablation.feature_group(f)
        sizes[g] = sizes.get(g, 0) + 1
    assert sizes == {"preco": 2, "cambio": 6, "brent": 6, "clima": 24,
                     "sensoriamento": 12, "calendario": 4}


def test_family_sizes_cover_all_features():
    fams = ablation.families_of(FEATURES)
    assert set(fams) == set(ablation.FAMILY_LABELS)
    assert sum(len(v) for v in fams.values()) == 54
    assert {k: len(v) for k, v in fams.items()} == {
        "preco_brl": 1, "preco_usd": 1, "cambio": 6, "brent": 6, "precipitacao": 6,
        "temperatura_media": 6, "temperatura_max": 1, "temperatura_min": 1,
        "umidade_relativa": 5, "vento": 5, "ndvi": 6, "umidade_solo": 6, "calendario": 4,
    }


def test_selection_rule():
    imp = pd.DataFrame({"delta_mae_mean": [1.0, 0.1, -0.2, 0.5],
                        "delta_mae_std": [0.2, 0.1, 0.05, 0.25]})
    assert ablation.select_by_rule(imp).tolist() == [True, False, False, False]


def test_configs_over_relevant_families():
    relevant = ["preco_brl", "cambio"]
    configs = ablation.build_configs(FEATURES, relevant)
    assert configs["todas_54"] == FEATURES
    rel = configs["relevantes"]
    assert {ablation.feature_family(f) for f in rel} == set(relevant)
    assert len(rel) == 7
    assert set(configs) == {"todas_54", "relevantes", "sem_preco_brl", "sem_cambio"}
    assert configs["sem_preco_brl"] == [f for f in rel if f != "soy_price_brl_bag"]
    assert all(ablation.feature_family(f) == "preco_brl" for f in configs["sem_cambio"])


def test_unknown_feature_raises():
    with pytest.raises(ValueError):
        ablation.feature_group("coluna_desconhecida")
    with pytest.raises(ValueError):
        ablation.feature_family("coluna_desconhecida")
