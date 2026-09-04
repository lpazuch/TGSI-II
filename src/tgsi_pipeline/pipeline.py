from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .csv_io import write_csv
from .models import ProjectConfig
from .sources import bcb, cepea, eia, inmet, nasa_power, remote_sensing, world_bank
from .transforms import attach_series, build_base_rows, resample_rows
from .utils import coalesce, daterange, ensure_directory, parse_date_any


@dataclass(slots=True)
class PipelineResult:
    raw_files: list[str] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_pipeline(
    config: ProjectConfig,
    *,
    allow_partial: bool = True,
    dry_run: bool = False,
) -> PipelineResult:
    result = PipelineResult()
    base_dir = ensure_directory(config.output_base_dir)
    raw_dir = ensure_directory(base_dir / "raw")
    processed_dir = ensure_directory(base_dir / "processed")

    if dry_run:
        _write_manifest(base_dir, result, dry_run=True)
        return result

    climate_rows: list[dict[str, Any]] = []
    ndvi_rows: list[dict[str, Any]] = []
    soil_moisture_rows: list[dict[str, Any]] = []

    if config.sources.climate.enabled:
        for location in config.locations:
            inmet_rows, nasa_rows, combined_rows = _fetch_climate_for_location(
                config,
                location,
                result,
                allow_partial,
            )
            if inmet_rows:
                path = raw_dir / f"climate_inmet_{location.id}.csv"
                write_csv(path, inmet_rows)
                result.raw_files.append(str(path))
            if nasa_rows:
                path = raw_dir / f"climate_nasa_power_{location.id}.csv"
                write_csv(path, nasa_rows)
                result.raw_files.append(str(path))
            if combined_rows:
                path = raw_dir / f"climate_combined_{location.id}.csv"
                write_csv(path, combined_rows)
                result.raw_files.append(str(path))
                climate_rows.extend(combined_rows)

    if config.sources.economics.enabled:
        fx_rows = _guard_fetch(
            result,
            allow_partial,
            "taxa de cambio PTAX/BCB",
            lambda: bcb.fetch_fx_series(
                config.date_range,
                currency=config.sources.economics.fx_currency,
            ),
        )
        if fx_rows:
            path = raw_dir / "economics_fx_bcb.csv"
            write_csv(path, fx_rows)
            result.raw_files.append(str(path))
    else:
        fx_rows = []

    if config.sources.economics.enabled:
        oil_rows = _fetch_oil_series(config, result, allow_partial)
        if oil_rows:
            suffix = oil_rows[0].get("oil_frequency", "unknown")
            path = raw_dir / f"economics_brent_{suffix}.csv"
            write_csv(path, oil_rows)
            result.raw_files.append(str(path))
    else:
        oil_rows = []

    if config.sources.remote_sensing.enabled:
        for location in config.locations:
            payload = _guard_fetch(
                result,
                allow_partial,
                f"sensoriamento remoto para {location.id}",
                lambda location=location: remote_sensing.fetch_remote_sensing(
                    location,
                    config.date_range,
                    config.sources.remote_sensing,
                ),
            )
            if not payload:
                continue
            ndvi_rows.extend(payload.get("ndvi", []))
            soil_moisture_rows.extend(payload.get("soil_moisture", []))

    if ndvi_rows:
        grouped = _group_rows_by_location(ndvi_rows)
        for location_id, rows in grouped.items():
            path = raw_dir / f"remote_ndvi_{location_id}.csv"
            write_csv(path, rows)
            result.raw_files.append(str(path))

    if soil_moisture_rows:
        grouped = _group_rows_by_location(soil_moisture_rows)
        for location_id, rows in grouped.items():
            path = raw_dir / f"remote_soil_moisture_{location_id}.csv"
            write_csv(path, rows)
            result.raw_files.append(str(path))

    if config.target.enabled and config.target.path:
        target_daily_rows, target_monthly_rows = _fetch_target_series(config, result, allow_partial)
        if target_daily_rows:
            path = raw_dir / "target_soy_cepea_daily.csv"
            write_csv(path, target_daily_rows)
            result.raw_files.append(str(path))
        if target_monthly_rows:
            path = raw_dir / "target_soy_cepea_monthly.csv"
            write_csv(path, target_monthly_rows)
            result.raw_files.append(str(path))
    else:
        target_daily_rows = []
        target_monthly_rows = []

    daily_rows = _build_daily_features(config, climate_rows, fx_rows, oil_rows, ndvi_rows, soil_moisture_rows)
    monthly_rows = _build_monthly_features(config, climate_rows, fx_rows, oil_rows, ndvi_rows, soil_moisture_rows)
    monthly_ml_rows = _build_monthly_ml_features(monthly_rows)
    monthly_ml_target_rows = _attach_monthly_target(monthly_ml_rows, target_monthly_rows)
    monthly_modeling_rows = _build_monthly_modeling_dataset(monthly_ml_target_rows)

    daily_path = processed_dir / "features_daily.csv"
    monthly_path = processed_dir / "features_monthly.csv"
    monthly_ml_path = processed_dir / "features_monthly_ml.csv"
    write_csv(daily_path, daily_rows)
    write_csv(monthly_path, monthly_rows)
    write_csv(monthly_ml_path, monthly_ml_rows)
    result.processed_files.extend([str(daily_path), str(monthly_path), str(monthly_ml_path)])
    if monthly_ml_target_rows:
        monthly_ml_target_path = processed_dir / "features_monthly_ml_target.csv"
        write_csv(monthly_ml_target_path, monthly_ml_target_rows)
        result.processed_files.append(str(monthly_ml_target_path))
    if monthly_modeling_rows:
        monthly_modeling_path = processed_dir / "features_monthly_modeling.csv"
        write_csv(monthly_modeling_path, monthly_modeling_rows)
        result.processed_files.append(str(monthly_modeling_path))

    _write_manifest(base_dir, result, dry_run=False)
    if result.errors and not allow_partial:
        raise RuntimeError("Falhas no pipeline impediram a execucao completa.")
    return result


def _fetch_climate_for_location(
    config: ProjectConfig,
    location: Any,
    result: PipelineResult,
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inmet_rows = _guard_fetch(
        result,
        allow_partial,
        f"clima INMET para {location.id}",
        lambda: inmet.fetch_daily_climate(location, config.date_range),
    ) or []

    needs_fallback = config.sources.climate.nasa_power_fallback
    nasa_rows: list[dict[str, Any]] = []
    if needs_fallback:
        nasa_rows = _guard_fetch(
            result,
            allow_partial,
            f"clima NASA POWER para {location.id}",
            lambda: nasa_power.fetch_daily_climate(
                location,
                config.date_range,
                config.sources.climate.nasa_power_parameters,
            ),
        ) or []

    combined_rows = _combine_climate_sources(location, inmet_rows, nasa_rows)
    return inmet_rows, nasa_rows, combined_rows


def _fetch_oil_series(
    config: ProjectConfig,
    result: PipelineResult,
    allow_partial: bool,
) -> list[dict[str, Any]]:
    provider = config.sources.economics.oil.provider.lower()
    frequency = config.sources.economics.oil.frequency.lower()

    if provider == "world_bank":
        return _guard_fetch(
            result,
            allow_partial,
            "Brent via World Bank",
            lambda: world_bank.fetch_brent_monthly(config.date_range),
        ) or []

    if provider == "eia":
        rows = _guard_fetch(
            result,
            allow_partial,
            f"Brent via EIA ({frequency})",
            lambda: eia.fetch_brent_series(config.date_range, frequency=frequency),
        )
        if rows:
            return rows
        fallback = _guard_fetch(
            result,
            allow_partial,
            "Brent fallback via World Bank",
            lambda: world_bank.fetch_brent_monthly(config.date_range),
        )
        return fallback or []

    result.warnings.append(f"Provider de Brent desconhecido: {provider}.")
    return []


def _build_daily_features(
    config: ProjectConfig,
    climate_rows: list[dict[str, Any]],
    fx_rows: list[dict[str, Any]],
    oil_rows: list[dict[str, Any]],
    ndvi_rows: list[dict[str, Any]],
    soil_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_rows = build_base_rows(config.date_range, config.locations, "daily")
    if climate_rows:
        base_rows = attach_series(
            base_rows,
            climate_rows,
            key_fields=["location_id"],
            value_fields=[
                "precipitation_mm",
                "temperature_mean_c",
                "temperature_max_c",
                "temperature_min_c",
                "relative_humidity_pct",
                "wind_speed_ms",
                "climate_source",
            ],
            exact=True,
        )

    if fx_rows:
        base_rows = attach_series(
            base_rows,
            fx_rows,
            key_fields=[],
            value_fields=["usd_brl", "fx_source"],
            exact=False,
            max_lag_days=7,
        )

    oil_frequency = str(oil_rows[0].get("oil_frequency")) if oil_rows else ""
    if oil_rows:
        prepared_oil = oil_rows
        if oil_frequency == "monthly":
            prepared_oil = _expand_monthly_series(oil_rows, config)
        base_rows = attach_series(
            base_rows,
            prepared_oil,
            key_fields=[],
            value_fields=["brent_usd_bbl", "oil_source"],
            exact=True if oil_frequency == "monthly" else False,
            max_lag_days=None if oil_frequency == "monthly" else 7,
        )

    if ndvi_rows:
        base_rows = attach_series(
            base_rows,
            ndvi_rows,
            key_fields=["location_id"],
            value_fields=["ndvi", "ndvi_source"],
            exact=False,
            max_lag_days=32,
        )

    if soil_rows:
        base_rows = attach_series(
            base_rows,
            soil_rows,
            key_fields=["location_id"],
            value_fields=["soil_moisture_m3m3", "soil_moisture_source"],
            exact=False,
            max_lag_days=5,
        )
    return base_rows


def _build_monthly_features(
    config: ProjectConfig,
    climate_rows: list[dict[str, Any]],
    fx_rows: list[dict[str, Any]],
    oil_rows: list[dict[str, Any]],
    ndvi_rows: list[dict[str, Any]],
    soil_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_rows = build_base_rows(config.date_range, config.locations, "monthly")

    if climate_rows:
        monthly_climate = resample_rows(
            climate_rows,
            frequency="monthly",
            group_keys=["location_id", "location_name", "latitude", "longitude"],
            aggregations={
                "precipitation_mm": "sum",
                "temperature_mean_c": "mean",
                "temperature_max_c": "max",
                "temperature_min_c": "min",
                "relative_humidity_pct": "mean",
                "wind_speed_ms": "mean",
            },
        )
        climate_source_rows = _monthly_last_value(climate_rows, ["location_id"], "climate_source")
        monthly_climate = attach_series(
            monthly_climate,
            climate_source_rows,
            key_fields=["location_id"],
            value_fields=["climate_source"],
            exact=True,
        )
        base_rows = attach_series(
            base_rows,
            monthly_climate,
            key_fields=["location_id"],
            value_fields=[
                "precipitation_mm",
                "temperature_mean_c",
                "temperature_max_c",
                "temperature_min_c",
                "relative_humidity_pct",
                "wind_speed_ms",
                "climate_source",
            ],
            exact=True,
        )

    if fx_rows:
        monthly_fx = resample_rows(
            fx_rows,
            frequency="monthly",
            group_keys=[],
            aggregations={"usd_brl": "mean"},
        )
        fx_source_rows = _monthly_last_value(fx_rows, [], "fx_source")
        monthly_fx = attach_series(
            monthly_fx,
            fx_source_rows,
            key_fields=[],
            value_fields=["fx_source"],
            exact=True,
        )
        base_rows = attach_series(
            base_rows,
            monthly_fx,
            key_fields=[],
            value_fields=["usd_brl", "fx_source"],
            exact=True,
        )

    if oil_rows:
        if str(oil_rows[0].get("oil_frequency")) == "monthly":
            monthly_oil = oil_rows
        else:
            monthly_oil = resample_rows(
                oil_rows,
                frequency="monthly",
                group_keys=[],
                aggregations={"brent_usd_bbl": "mean"},
            )
            oil_source_rows = _monthly_last_value(oil_rows, [], "oil_source")
            monthly_oil = attach_series(
                monthly_oil,
                oil_source_rows,
                key_fields=[],
                value_fields=["oil_source"],
                exact=True,
            )
        base_rows = attach_series(
            base_rows,
            monthly_oil,
            key_fields=[],
            value_fields=["brent_usd_bbl", "oil_source"],
            exact=True,
        )

    if ndvi_rows:
        monthly_ndvi = resample_rows(
            ndvi_rows,
            frequency="monthly",
            group_keys=["location_id"],
            aggregations={"ndvi": "mean"},
        )
        ndvi_source_rows = _monthly_last_value(ndvi_rows, ["location_id"], "ndvi_source")
        monthly_ndvi = attach_series(
            monthly_ndvi,
            ndvi_source_rows,
            key_fields=["location_id"],
            value_fields=["ndvi_source"],
            exact=True,
        )
        base_rows = attach_series(
            base_rows,
            monthly_ndvi,
            key_fields=["location_id"],
            value_fields=["ndvi", "ndvi_source"],
            exact=True,
        )

    if soil_rows:
        monthly_soil = resample_rows(
            soil_rows,
            frequency="monthly",
            group_keys=["location_id"],
            aggregations={"soil_moisture_m3m3": "mean"},
        )
        soil_source_rows = _monthly_last_value(
            soil_rows,
            ["location_id"],
            "soil_moisture_source",
        )
        monthly_soil = attach_series(
            monthly_soil,
            soil_source_rows,
            key_fields=["location_id"],
            value_fields=["soil_moisture_source"],
            exact=True,
        )
        base_rows = attach_series(
            base_rows,
            monthly_soil,
            key_fields=["location_id"],
            value_fields=["soil_moisture_m3m3", "soil_moisture_source"],
            exact=True,
        )

    return base_rows


def _combine_climate_sources(
    location: Any,
    inmet_rows: list[dict[str, Any]],
    nasa_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_date: dict[str, dict[str, Any]] = {}
    for row in nasa_rows:
        rows_by_date[str(row["date"])] = dict(row)
    for row in inmet_rows:
        existing = rows_by_date.get(str(row["date"]))
        if existing is None:
            rows_by_date[str(row["date"])] = dict(row)
            continue
        merged = dict(existing)
        fields = [
            "precipitation_mm",
            "temperature_mean_c",
            "temperature_max_c",
            "temperature_min_c",
            "relative_humidity_pct",
            "wind_speed_ms",
        ]
        used_inmet = False
        used_nasa = False
        for field in fields:
            merged[field] = coalesce(row.get(field), existing.get(field))
            if row.get(field) is not None:
                used_inmet = True
            if existing.get(field) is not None and row.get(field) is None:
                used_nasa = True
        merged["location_id"] = location.id
        merged["location_name"] = location.name
        merged["latitude"] = location.latitude
        merged["longitude"] = location.longitude
        if used_inmet and used_nasa:
            merged["climate_source"] = "INMET+NASA_POWER"
        elif used_inmet:
            merged["climate_source"] = "INMET"
        else:
            merged["climate_source"] = "NASA_POWER"
        rows_by_date[str(row["date"])] = merged

    output = [rows_by_date[key] for key in sorted(rows_by_date)]
    for row in output:
        row.setdefault("location_id", location.id)
        row.setdefault("location_name", location.name)
        row.setdefault("latitude", location.latitude)
        row.setdefault("longitude", location.longitude)
    return output


def _expand_monthly_series(
    rows: list[dict[str, Any]],
    config: ProjectConfig,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    rows_by_month = {str(row["date"]): row for row in rows}
    for current_date in daterange(config.date_range.start, config.date_range.end):
        day = current_date.isoformat()
        month_key = day[:7] + "-01"
        source = rows_by_month.get(month_key)
        if not source:
            continue
        expanded.append(
            {
                "date": day,
                "brent_usd_bbl": source.get("brent_usd_bbl"),
                "oil_source": source.get("oil_source"),
            }
        )
    return expanded


def _monthly_last_value(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    field_name: str,
) -> list[dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: tuple(item.get(key) for key in group_keys) + (str(item["date"]),)):
        month = str(row["date"])[:7] + "-01"
        key = tuple(row.get(key) for key in group_keys) + (month,)
        payload = {key_name: row.get(key_name) for key_name in group_keys}
        payload["date"] = month
        payload[field_name] = row.get(field_name)
        output[key] = payload
    return [output[key] for key in sorted(output)]


def _group_rows_by_location(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["location_id"]), []).append(row)
    return grouped


def _fetch_target_series(
    config: ProjectConfig,
    result: PipelineResult,
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if config.target.source.lower() != "cepea_xls":
        result.warnings.append(f"Fonte de alvo desconhecida: {config.target.source}.")
        return [], []

    daily_rows = _guard_fetch(
        result,
        allow_partial,
        "alvo CEPEA da soja",
        lambda: cepea.fetch_soybean_prices_from_xls(
            config.target.path or "",
            price_column_brl=config.target.price_column_brl,
            price_column_usd=config.target.price_column_usd,
            series_name=config.target.series_name,
        ),
    ) or []
    monthly_rows = cepea.resample_monthly(daily_rows) if daily_rows else []
    return daily_rows, monthly_rows


def _attach_monthly_target(
    feature_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not feature_rows or not target_rows:
        return []

    target_by_date = {str(row["date"]): row for row in target_rows}
    ordered_target_dates = sorted(target_by_date)
    next_month_by_date: dict[str, dict[str, Any]] = {}
    for index, current_date in enumerate(ordered_target_dates[:-1]):
        next_month_by_date[current_date] = target_by_date[ordered_target_dates[index + 1]]

    output: list[dict[str, Any]] = []
    for row in feature_rows:
        current = dict(row)
        target = target_by_date.get(str(row["date"]))
        if target:
            current["soy_price_brl_bag"] = target.get("soy_price_brl_bag")
            current["soy_price_usd_bag"] = target.get("soy_price_usd_bag")
            current["target_source"] = target.get("target_source")
            current["target_series_name"] = target.get("target_series_name")
        next_target = next_month_by_date.get(str(row["date"]))
        current["soy_price_brl_bag_next_month"] = next_target.get("soy_price_brl_bag") if next_target else None
        current["soy_price_usd_bag_next_month"] = next_target.get("soy_price_usd_bag") if next_target else None
        output.append(current)
    return output


def _build_monthly_modeling_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    feature_candidates = [
        "month",
        "quarter",
        "soy_planting_window",
        "soy_harvest_window",
        "usd_brl",
        "brent_usd_bbl",
        "precipitation_mm",
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "relative_humidity_pct",
        "wind_speed_ms",
        "ndvi",
        "soil_moisture_m3m3",
        "usd_brl_lag_1",
        "usd_brl_lag_2",
        "usd_brl_lag_3",
        "usd_brl_rolling_mean_3",
        "brent_usd_bbl_lag_1",
        "brent_usd_bbl_lag_2",
        "brent_usd_bbl_lag_3",
        "brent_usd_bbl_rolling_mean_3",
        "precipitation_mm_lag_1",
        "precipitation_mm_lag_2",
        "precipitation_mm_lag_3",
        "precipitation_mm_rolling_mean_3",
        "temperature_mean_c_lag_1",
        "temperature_mean_c_lag_2",
        "temperature_mean_c_lag_3",
        "temperature_mean_c_rolling_mean_3",
        "relative_humidity_pct_lag_1",
        "relative_humidity_pct_lag_2",
        "relative_humidity_pct_lag_3",
        "relative_humidity_pct_rolling_mean_3",
        "wind_speed_ms_lag_1",
        "wind_speed_ms_lag_2",
        "wind_speed_ms_lag_3",
        "wind_speed_ms_rolling_mean_3",
        "ndvi_lag_1",
        "ndvi_lag_2",
        "ndvi_lag_3",
        "ndvi_rolling_mean_3",
        "soil_moisture_m3m3_lag_1",
        "soil_moisture_m3m3_lag_2",
        "soil_moisture_m3m3_lag_3",
        "soil_moisture_m3m3_rolling_mean_3",
        "usd_brl_month_anomaly",
        "brent_usd_bbl_month_anomaly",
        "precipitation_mm_month_anomaly",
        "temperature_mean_c_month_anomaly",
        "ndvi_month_anomaly",
        "soil_moisture_m3m3_month_anomaly",
        "soy_price_brl_bag",
        "soy_price_usd_bag",
    ]

    available_feature_candidates = [
        field_name for field_name in feature_candidates if any(row.get(field_name) not in (None, "") for row in rows)
    ]

    output: list[dict[str, Any]] = []
    for row in rows:
        target_value = row.get("soy_price_brl_bag_next_month")
        if target_value in (None, ""):
            continue
        current = {
            "date": row.get("date"),
            "target_month": _next_month_start(str(row.get("date"))),
            "target_variable": "soy_price_brl_bag_next_month",
            "target_value": target_value,
            "target_source": row.get("target_source"),
            "target_series_name": row.get("target_series_name"),
        }
        for field_name in available_feature_candidates:
            current[field_name] = row.get(field_name)
        output.append(current)
    return output


def _next_month_start(date_value: str) -> str:
    parsed = parse_date_any(date_value)
    year = parsed.year + (1 if parsed.month == 12 else 0)
    month = 1 if parsed.month == 12 else parsed.month + 1
    return f"{year:04d}-{month:02d}-01"


def _build_monthly_ml_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    ordered_rows = sorted(
        (dict(row) for row in rows),
        key=lambda item: (str(item.get("location_id", "")), str(item["date"])),
    )
    anomaly_fields = [
        "precipitation_mm",
        "temperature_mean_c",
        "ndvi",
        "soil_moisture_m3m3",
        "usd_brl",
        "brent_usd_bbl",
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered_rows:
        grouped[str(row.get("location_id", ""))].append(row)

    output: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        previous_rows: list[dict[str, Any]] = []
        for row in group_rows:
            current = dict(row)
            current_date = parse_date_any(str(current["date"]))
            month = current_date.month
            current["month"] = month
            current["quarter"] = ((month - 1) // 3) + 1
            current["soy_planting_window"] = 1 if month in {9, 10, 11, 12} else 0
            current["soy_harvest_window"] = 1 if month in {1, 2, 3, 4} else 0

            for field in [
                "precipitation_mm",
                "temperature_mean_c",
                "relative_humidity_pct",
                "wind_speed_ms",
                "ndvi",
                "soil_moisture_m3m3",
                "usd_brl",
                "brent_usd_bbl",
            ]:
                values = [item.get(field) for item in previous_rows]
                current[f"{field}_lag_1"] = _lag(values, 1)
                current[f"{field}_lag_2"] = _lag(values, 2)
                current[f"{field}_lag_3"] = _lag(values, 3)
                current[f"{field}_rolling_mean_3"] = _rolling_mean(values, 3)

            # Anomalia mensal CAUSAL (sem look-ahead):
            #   anomaly(t) = valor(t) - media{ valor(s) : s < t, mes(s) == mes(t) }
            # A "normal" do mes e recalculada a cada linha usando apenas o passado
            # (previous_rows), exatamente como um walk-forward de janela expansiva
            # faria em cada passo. Nas primeiras ocorrencias de cada mes-calendario
            # (sem historico daquele mes) a anomalia fica None.
            # Ver docs/methodology.md, secao "Anomalias e vazamento temporal".
            for field in anomaly_fields:
                value = current.get(field)
                normal = _causal_month_normal(previous_rows, month, field)
                current[f"{field}_month_anomaly"] = (
                    round(value - normal, 6)
                    if isinstance(value, (int, float)) and normal is not None
                    else None
                )

            output.append(current)
            previous_rows.append(row)
    return output


def _causal_month_normal(
    previous_rows: list[dict[str, Any]],
    month: int,
    field: str,
) -> float | None:
    """Media de ``field`` sobre as linhas passadas do mesmo mes-calendario.

    Usa somente ``previous_rows`` (datas estritamente anteriores a linha atual),
    garantindo que a referencia historica nao contenha informacao futura.
    """
    values = [
        float(row[field])
        for row in previous_rows
        if parse_date_any(str(row["date"])).month == month
        and isinstance(row.get(field), (int, float))
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _lag(values: list[Any], steps: int) -> Any:
    if len(values) < steps:
        return None
    return values[-steps]


def _rolling_mean(values: list[Any], window: int) -> float | None:
    numeric = [float(value) for value in values[-window:] if isinstance(value, (int, float))]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def _guard_fetch(
    result: PipelineResult,
    allow_partial: bool,
    label: str,
    fetcher: Any,
) -> Any:
    try:
        return fetcher()
    except Exception as exc:  # noqa: BLE001
        message = f"Falha em {label}: {exc}"
        if allow_partial:
            result.warnings.append(message)
            return None
        result.errors.append(message)
        raise


def _write_manifest(base_dir: Path, result: PipelineResult, *, dry_run: bool) -> None:
    manifest = {
        "dry_run": dry_run,
        "raw_files": result.raw_files,
        "processed_files": result.processed_files,
        "warnings": result.warnings,
        "errors": result.errors,
    }
    path = base_dir / "run_report.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
