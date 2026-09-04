from __future__ import annotations

from typing import Any

from ..http import add_query_params, request_json
from ..models import DateRange, Location
from ..utils import parse_float


NASA_POWER_DAILY_POINT = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_daily_climate(
    location: Location,
    date_range: DateRange,
    parameters: list[str],
) -> list[dict[str, Any]]:
    url = add_query_params(
        NASA_POWER_DAILY_POINT,
        {
            "parameters": ",".join(parameters),
            "community": "AG",
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start": date_range.start.strftime("%Y%m%d"),
            "end": date_range.end.strftime("%Y%m%d"),
            "format": "JSON",
        },
    )
    payload = request_json(url)
    series = payload.get("properties", {}).get("parameter", {})
    if not isinstance(series, dict) or not series:
        raise ValueError("Resposta inesperada da NASA POWER.")

    date_keys: set[str] = set()
    for values in series.values():
        if isinstance(values, dict):
            date_keys.update(values.keys())

    rows: list[dict[str, Any]] = []
    for raw_date in sorted(date_keys):
        precipitation = _clean_value(series.get("PRECTOTCORR", {}).get(raw_date))
        temperature_mean = _clean_value(series.get("T2M", {}).get(raw_date))
        temperature_max = _clean_value(series.get("T2M_MAX", {}).get(raw_date))
        temperature_min = _clean_value(series.get("T2M_MIN", {}).get(raw_date))
        relative_humidity = _clean_value(series.get("RH2M", {}).get(raw_date))
        wind_speed = _clean_value(series.get("WS2M", {}).get(raw_date))

        rows.append(
            {
                "date": f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}",
                "location_id": location.id,
                "location_name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "precipitation_mm": precipitation,
                "temperature_mean_c": temperature_mean,
                "temperature_max_c": temperature_max,
                "temperature_min_c": temperature_min,
                "relative_humidity_pct": relative_humidity,
                "wind_speed_ms": wind_speed,
                "climate_source": "NASA_POWER",
            }
        )
    return rows


def _clean_value(value: object) -> float | None:
    number = parse_float(value)
    if number is None:
        return None
    if number <= -999:
        return None
    return number
