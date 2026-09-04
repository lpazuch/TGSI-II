from __future__ import annotations

import csv
import io
from collections import defaultdict
from statistics import mean
from typing import Any
from zipfile import ZipFile

from ..http import HttpRequestError, request_bytes
from ..models import DateRange, Location
from ..utils import normalize_text, parse_date_any, parse_float


INMET_ZIP_URL_TEMPLATE = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"


def fetch_daily_climate(
    location: Location,
    date_range: DateRange,
) -> list[dict[str, Any]]:
    station_rows: list[dict[str, Any]] = []
    for year in date_range.years():
        try:
            archive = request_bytes(INMET_ZIP_URL_TEMPLATE.format(year=year), timeout=240)
        except HttpRequestError:
            continue
        station_rows.extend(_extract_station_rows_from_archive(archive, location, date_range))
    return _merge_station_rows(station_rows)


def _extract_station_rows_from_archive(
    archive_bytes: bytes,
    location: Location,
    date_range: DateRange,
) -> list[dict[str, Any]]:
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        ]
        selected = _select_station_members(archive, members, location)
        rows: list[dict[str, Any]] = []
        for member in selected:
            raw_bytes = archive.read(member)
            text = _decode_csv(raw_bytes)
            rows.extend(_parse_station_csv(text, location, date_range))
        return rows


def _select_station_members(
    archive: ZipFile,
    members: list[str],
    location: Location,
) -> list[str]:
    if not members:
        return []

    if location.inmet_station_codes:
        selected = []
        normalized_members = {member: normalize_text(member) for member in members}
        for code in location.inmet_station_codes:
            code_normalized = normalize_text(code)
            for member, member_normalized in normalized_members.items():
                if code_normalized in member_normalized:
                    selected.append(member)
        if selected:
            return sorted(set(selected))

        selected_by_metadata = []
        for member in members:
            metadata = _peek_metadata(_decode_csv(archive.read(member)))
            metadata_probe = " ".join(
                normalize_text(value)
                for value in (
                    metadata.get("codigo (wmo)", ""),
                    metadata.get("codigo", ""),
                    metadata.get("estacao", ""),
                )
            )
            for code in location.inmet_station_codes:
                if normalize_text(code) in metadata_probe:
                    selected_by_metadata.append(member)
                    break
        if selected_by_metadata:
            return sorted(set(selected_by_metadata))

    location_probe = normalize_text(location.name)
    selected = [member for member in members if location_probe in normalize_text(member)]
    if selected:
        return sorted(set(selected))

    return []


def _decode_csv(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _parse_station_csv(
    text: str,
    location: Location,
    date_range: DateRange,
) -> list[dict[str, Any]]:
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    header_index = _find_header_index(rows)
    if header_index is None:
        return []

    metadata = _parse_metadata(rows[:header_index])
    header = rows[header_index]
    data_rows = rows[header_index + 1 :]
    columns = _resolve_columns(header)
    if columns["date"] is None:
        return []

    daily: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for values in data_rows:
        if not any(cell.strip() for cell in values):
            continue
        padded = values + [""] * (len(header) - len(values))
        record = dict(zip(header, padded))
        raw_date = record.get(columns["date"] or "", "").strip()
        if not raw_date:
            continue
        try:
            current_date = parse_date_any(raw_date)
        except ValueError:
            continue
        if not (date_range.start <= current_date <= date_range.end):
            continue

        bucket = daily[current_date.isoformat()]
        _append_metric(bucket["precipitation_mm"], record, columns["precipitation_mm"])
        _append_metric(bucket["temperature_mean_c"], record, columns["temperature_mean_c"])
        _append_metric(bucket["temperature_max_c"], record, columns["temperature_max_c"])
        _append_metric(bucket["temperature_min_c"], record, columns["temperature_min_c"])
        _append_metric(bucket["relative_humidity_pct"], record, columns["relative_humidity_pct"])
        _append_metric(bucket["wind_speed_ms"], record, columns["wind_speed_ms"])

    station_code = metadata.get("codigo (wmo)") or metadata.get("codigo")
    station_name = metadata.get("estacao")
    output: list[dict[str, Any]] = []
    for day in sorted(daily):
        metrics = daily[day]
        output.append(
            {
                "date": day,
                "location_id": location.id,
                "location_name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "precipitation_mm": sum(metrics["precipitation_mm"]) if metrics["precipitation_mm"] else None,
                "temperature_mean_c": mean(metrics["temperature_mean_c"]) if metrics["temperature_mean_c"] else None,
                "temperature_max_c": max(metrics["temperature_max_c"]) if metrics["temperature_max_c"] else None,
                "temperature_min_c": min(metrics["temperature_min_c"]) if metrics["temperature_min_c"] else None,
                "relative_humidity_pct": mean(metrics["relative_humidity_pct"]) if metrics["relative_humidity_pct"] else None,
                "wind_speed_ms": mean(metrics["wind_speed_ms"]) if metrics["wind_speed_ms"] else None,
                "climate_source": "INMET",
                "inmet_station_code": station_code,
                "inmet_station_name": station_name,
            }
        )
    return output


def _find_header_index(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows):
        normalized = [normalize_text(cell) for cell in row]
        if any("data medicao" in cell for cell in normalized):
            return index
    return None


def _parse_metadata(rows: list[list[str]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for row in rows:
        if len(row) < 2:
            continue
        key = normalize_text(row[0])
        value = row[1].strip()
        if key and value:
            metadata[key] = value
    return metadata


def _peek_metadata(text: str) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    header_index = _find_header_index(rows)
    if header_index is None:
        return {}
    return _parse_metadata(rows[:header_index])


def _resolve_columns(header: list[str]) -> dict[str, str | None]:
    normalized = {column: normalize_text(column) for column in header}

    def pick(*predicates: str) -> str | None:
        for column, probe in normalized.items():
            if all(predicate in probe for predicate in predicates):
                return column
        return None

    return {
        "date": pick("data", "medicao") or pick("data"),
        "precipitation_mm": pick("precipitacao", "horario"),
        "temperature_mean_c": pick("temperatura do ar", "bulbo seco")
        or pick("bulbo seco", "horaria"),
        "temperature_max_c": pick("temperatura maxima", "hora ant"),
        "temperature_min_c": pick("temperatura minima", "hora ant"),
        "relative_humidity_pct": pick("umidade relativa do ar", "horaria")
        or pick("umidade relativa", "horaria"),
        "wind_speed_ms": pick("vento", "velocidade horaria")
        or pick("velocidade do vento"),
    }


def _append_metric(bucket: list[float], record: dict[str, str], column: str | None) -> None:
    if not column:
        return
    value = parse_float(record.get(column))
    if value is not None:
        bucket.append(value)


def _merge_station_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_fields = [
        "precipitation_mm",
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "relative_humidity_pct",
        "wind_speed_ms",
    ]
    for row in sorted(rows, key=lambda item: (item["location_id"], item["date"])):
        key = (str(row["location_id"]), str(row["date"]))
        if key not in merged:
            merged[key] = dict(row)
            continue
        for field in ordered_fields:
            if merged[key].get(field) is None and row.get(field) is not None:
                merged[key][field] = row.get(field)
        if not merged[key].get("inmet_station_code") and row.get("inmet_station_code"):
            merged[key]["inmet_station_code"] = row.get("inmet_station_code")
        if not merged[key].get("inmet_station_name") and row.get("inmet_station_name"):
            merged[key]["inmet_station_name"] = row.get("inmet_station_name")
    return [merged[key] for key in sorted(merged)]
