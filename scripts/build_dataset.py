#!/usr/bin/env python3
"""
build_dataset.py — reconstroi a base de modelagem a partir dos dados de entrada.

Fluxo:

    dados brutos (data/raw/*.csv, CEPEA_*.xls)
        -> leitura / coercao de tipos
        -> integracao das fontes            (tgsi_pipeline._build_monthly_features)
        -> agregacao mensal                 (idem)
        -> features derivadas               (tgsi_pipeline._build_monthly_ml_features)
             month/quarter, lags 1-3, media movel 3, anomalias mensais CAUSAIS
        -> anexo do alvo (t -> t+1)         (tgsi_pipeline._attach_monthly_target)
        -> dataset final                    (tgsi_pipeline._build_monthly_modeling_dataset)
        -> data/processed/features_monthly_modeling.csv

Dois modos:

  --offline  (padrao): NAO acessa a rede. Le os CSVs ja presentes em data/raw/
             e roda apenas as transformacoes. E o caminho de reproducao para
             quem recebe o codigo + os arquivos de entrada (nao precisa de
             credenciais NASA Earthdata nem de horas de espera no AppEEARS).

  --online : re-baixa tudo das fontes originais (BCB, EIA, NASA POWER, INMET,
             AppEEARS) conforme configs/dataset.hybrid_2006plus.json. Exige rede
             e, para o sensoriamento remoto, credenciais EARTHDATA_USERNAME /
             EARTHDATA_PASSWORD.

Uso:
  PYTHONPATH=src python scripts/build_dataset.py
  PYTHONPATH=src python scripts/build_dataset.py --online
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tgsi_pipeline.config import load_config
from tgsi_pipeline.csv_io import read_csv, write_csv
from tgsi_pipeline.pipeline import (
    _attach_monthly_target,
    _build_daily_features,
    _build_monthly_features,
    _build_monthly_ml_features,
    _build_monthly_modeling_dataset,
    run_pipeline,
)
from tgsi_pipeline.utils import parse_float

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.hybrid_2006plus.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Colunas que devem permanecer como texto (todo o resto vira float quando possivel).
_STRING_FIELDS = {
    "date",
    "location_id",
    "location_name",
    "climate_source",
    "fx_source",
    "oil_source",
    "oil_frequency",
    "ndvi_source",
    "soil_moisture_source",
    "inmet_station_code",
    "inmet_station_name",
    "target_source",
    "target_series_name",
    "target_frequency",
}

# raw CSV -> nome logico usado pelas transformacoes do pipeline
_RAW_INPUTS = {
    "climate_rows": "climate_combined_sorriso_mt.csv",
    "fx_rows": "economics_fx_bcb.csv",
    "oil_rows": "economics_brent_daily.csv",
    "ndvi_rows": "remote_ndvi_sorriso_mt.csv",
    "soil_moisture_rows": "remote_soil_moisture_sorriso_mt.csv",
}


def _coerce_row(row: dict) -> dict:
    out: dict = {}
    for key, value in row.items():
        if key in _STRING_FIELDS:
            out[key] = value if value not in ("", None) else None
            continue
        number = parse_float(value)
        out[key] = number if number is not None else (value or None)
    return out


def _load_rows(name: str) -> list[dict]:
    path = RAW_DIR / name
    if not path.exists():
        return []
    return [_coerce_row(row) for row in read_csv(path)]


def build_offline(config_path: Path) -> Path:
    config = load_config(config_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    inputs = {logical: _load_rows(fname) for logical, fname in _RAW_INPUTS.items()}
    missing = [fname for logical, fname in _RAW_INPUTS.items() if not inputs[logical]]
    if inputs["fx_rows"] == [] or inputs["oil_rows"] == []:
        raise SystemExit(
            "Faltam arquivos de entrada obrigatorios em data/raw/. "
            f"Ausentes/vazios: {missing}\n"
            "Coloque os CSVs de entrada em data/raw/ ou rode com --online."
        )
    if missing:
        print(f"[aviso] entradas ausentes (seguindo sem elas): {missing}")

    target_monthly = _load_rows_target()

    print("Integrando fontes e agregando por mes...")
    daily_rows = _build_daily_features(
        config,
        inputs["climate_rows"],
        inputs["fx_rows"],
        inputs["oil_rows"],
        inputs["ndvi_rows"],
        inputs["soil_moisture_rows"],
    )
    monthly_rows = _build_monthly_features(
        config,
        inputs["climate_rows"],
        inputs["fx_rows"],
        inputs["oil_rows"],
        inputs["ndvi_rows"],
        inputs["soil_moisture_rows"],
    )
    print("Gerando features derivadas (lags, media movel, anomalias causais)...")
    monthly_ml_rows = _build_monthly_ml_features(monthly_rows)
    monthly_ml_target_rows = _attach_monthly_target(monthly_ml_rows, target_monthly)
    modeling_rows = _build_monthly_modeling_dataset(monthly_ml_target_rows)

    write_csv(PROCESSED_DIR / "features_daily.csv", daily_rows)
    write_csv(PROCESSED_DIR / "features_monthly.csv", monthly_rows)
    write_csv(PROCESSED_DIR / "features_monthly_ml.csv", monthly_ml_rows)
    write_csv(PROCESSED_DIR / "features_monthly_ml_target.csv", monthly_ml_target_rows)
    out_path = PROCESSED_DIR / "features_monthly_modeling.csv"
    write_csv(out_path, modeling_rows)
    return out_path


def _load_rows_target() -> list[dict]:
    rows = _load_rows("target_soy_cepea_monthly.csv")
    if not rows:
        raise SystemExit(
            "data/raw/target_soy_cepea_monthly.csv nao encontrado.\n"
            "Ele e derivado da planilha CEPEA (data/raw/CEPEA_*.xls) pelo modo --online, "
            "ou deve ser recebido junto com o codigo."
        )
    return rows


def build_online(config_path: Path) -> Path:
    config = load_config(config_path)
    print(f"Config: {config_path.name} | janela {config.date_range.start} -> {config.date_range.end}")
    result = run_pipeline(config, allow_partial=True, dry_run=False)
    for warning in result.warnings:
        print(f"[aviso] {warning}")
    for error in result.errors:
        print(f"[ERRO] {error}")
    out_path = PROCESSED_DIR / "features_monthly_modeling.csv"
    if not out_path.exists():
        raise SystemExit("Pipeline online terminou sem gerar features_monthly_modeling.csv.")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstroi data/processed/features_monthly_modeling.csv")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", dest="offline", action="store_true", default=True,
                      help="Reconstroi so a partir de data/raw/ (padrao).")
    mode.add_argument("--online", dest="offline", action="store_false",
                      help="Re-baixa tudo das fontes originais (exige rede/credenciais).")
    args = parser.parse_args()

    print("=" * 62)
    print("  TGSI II — construcao da base de modelagem")
    print(f"  modo: {'offline (data/raw -> data/processed)' if args.offline else 'online (fontes originais)'}")
    print("=" * 62)

    out_path = build_offline(args.config) if args.offline else build_online(args.config)

    rows = read_csv(out_path)
    dates = sorted(r["date"] for r in rows)
    print("\nOK.")
    print(f"  {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(rows)} observacoes | {dates[0]} -> {dates[-1]}")
    print(f"  alvo: {rows[0].get('target_variable')} (t -> t+1)")


if __name__ == "__main__":
    main()
