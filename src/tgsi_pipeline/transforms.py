from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import mean
from typing import Any

from .models import DateRange, Location
from .utils import daterange, month_start, parse_date_any


def build_base_rows(
    date_range: DateRange,
    locations: list[Location],
    frequency: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frequency == "daily":
        dates = list(daterange(date_range.start, date_range.end))
    elif frequency == "monthly":
        seen: set[date] = set()
        dates = []
        for day in daterange(date_range.start, date_range.end):
            bucket = month_start(day)
            if bucket not in seen:
                seen.add(bucket)
                dates.append(bucket)
    else:
        raise ValueError(f"Frequencia invalida: {frequency}")

    for location in locations:
        for current in dates:
            rows.append(
                {
                    "date": current.isoformat(),
                    "location_id": location.id,
                    "location_name": location.name,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                }
            )
    return rows


def resample_rows(
    rows: list[dict[str, Any]],
    *,
    frequency: str,
    group_keys: list[str],
    aggregations: dict[str, str],
) -> list[dict[str, Any]]:
    if frequency not in {"daily", "monthly"}:
        raise ValueError(f"Frequencia invalida: {frequency}")

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    values_map: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        row_date = parse_date_any(str(row["date"]))
        bucket = row_date if frequency == "daily" else month_start(row_date)
        key = tuple(row.get(field) for field in group_keys) + (bucket.isoformat(),)
        if key not in grouped:
            grouped[key] = {field: row.get(field) for field in group_keys}
            grouped[key]["date"] = bucket.isoformat()

        for field, mode in aggregations.items():
            value = row.get(field)
            if value is None:
                continue
            if mode == "last":
                values_map[key][field] = [float(value)]
            else:
                values_map[key][field].append(float(value))

    output: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        row = dict(grouped[key])
        for field, mode in aggregations.items():
            values = values_map[key].get(field, [])
            if not values:
                row[field] = None
            elif mode == "sum":
                row[field] = round(sum(values), 6)
            elif mode == "mean":
                row[field] = round(mean(values), 6)
            elif mode == "min":
                row[field] = round(min(values), 6)
            elif mode == "max":
                row[field] = round(max(values), 6)
            elif mode == "last":
                row[field] = round(values[-1], 6)
            else:
                raise ValueError(f"Agregacao invalida: {mode}")
        output.append(row)
    return output


def attach_series(
    base_rows: list[dict[str, Any]],
    series_rows: list[dict[str, Any]],
    *,
    key_fields: list[str],
    value_fields: list[str],
    exact: bool,
    max_lag_days: int | None = None,
) -> list[dict[str, Any]]:
    grouped_series: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in series_rows:
        group = tuple(row.get(field) for field in key_fields)
        grouped_series[group].append(row)

    for series in grouped_series.values():
        series.sort(key=lambda item: parse_date_any(str(item["date"])))

    output: list[dict[str, Any]] = []
    state: dict[tuple[Any, ...], tuple[int, dict[str, Any] | None]] = {}

    for row in sorted(
        base_rows,
        key=lambda item: tuple(item.get(field) for field in key_fields)
        + (parse_date_any(str(item["date"])),),
    ):
        group = tuple(row.get(field) for field in key_fields)
        current_date = parse_date_any(str(row["date"]))
        series = grouped_series.get(group, [])
        index, current = state.get(group, (0, None))

        while index < len(series):
            candidate = series[index]
            candidate_date = parse_date_any(str(candidate["date"]))
            if candidate_date > current_date:
                break
            current = candidate
            index += 1

        state[group] = (index, current)
        merged = dict(row)

        if current is None:
            for field in value_fields:
                merged[field] = None
            output.append(merged)
            continue

        current_date_series = parse_date_any(str(current["date"]))
        lag_days = (current_date - current_date_series).days
        can_use = current_date_series == current_date if exact else True
        if max_lag_days is not None and lag_days > max_lag_days:
            can_use = False

        for field in value_fields:
            merged[field] = current.get(field) if can_use else None
        output.append(merged)

    return output
