from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import (
    ClimateConfig,
    DateRange,
    EconomicsConfig,
    Location,
    OilConfig,
    ProjectConfig,
    RemoteSensingConfig,
    SourcesConfig,
    TargetConfig,
)


def _parse_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    date_range = DateRange(
        start=_parse_date(raw["date_range"]["start"]),
        end=_parse_date(raw["date_range"]["end"]),
    )

    locations = [
        Location(
            id=item["id"],
            name=item["name"],
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            inmet_station_codes=list(item.get("inmet_station_codes", [])),
        )
        for item in raw["locations"]
    ]

    climate_raw = raw.get("sources", {}).get("climate", {})
    remote_raw = raw.get("sources", {}).get("remote_sensing", {})
    economics_raw = raw.get("sources", {}).get("economics", {})
    oil_raw = economics_raw.get("oil", {})
    target_raw = raw.get("target", {})

    sources = SourcesConfig(
        climate=ClimateConfig(
            enabled=bool(climate_raw.get("enabled", True)),
            nasa_power_fallback=bool(climate_raw.get("nasa_power_fallback", True)),
            nasa_power_parameters=list(
                climate_raw.get(
                    "nasa_power_parameters",
                    [
                        "PRECTOTCORR",
                        "T2M",
                        "T2M_MAX",
                        "T2M_MIN",
                        "RH2M",
                        "WS2M",
                    ],
                )
            ),
        ),
        remote_sensing=RemoteSensingConfig(
            enabled=bool(remote_raw.get("enabled", False)),
            earthdata_username_env=remote_raw.get(
                "earthdata_username_env", "EARTHDATA_USERNAME"
            ),
            earthdata_password_env=remote_raw.get(
                "earthdata_password_env", "EARTHDATA_PASSWORD"
            ),
            ndvi_product_id=remote_raw.get("ndvi_product_id", "MOD13Q1.061"),
            ndvi_layer=remote_raw.get("ndvi_layer"),
            soil_moisture_product_id=remote_raw.get(
                "soil_moisture_product_id", "SPL3SMP_E.006"
            ),
            soil_moisture_layer=remote_raw.get("soil_moisture_layer"),
            poll_interval_seconds=int(remote_raw.get("poll_interval_seconds", 15)),
            task_timeout_seconds=int(remote_raw.get("task_timeout_seconds", 1800)),
        ),
        economics=EconomicsConfig(
            enabled=bool(economics_raw.get("enabled", True)),
            fx_currency=economics_raw.get("fx_currency", "USD"),
            oil=OilConfig(
                provider=oil_raw.get("provider", "eia"),
                frequency=oil_raw.get("frequency", "daily"),
            ),
        ),
    )

    return ProjectConfig(
        date_range=date_range,
        target_frequencies=list(raw.get("target_frequencies", ["daily", "monthly"])),
        output_base_dir=Path(raw.get("output_base_dir", "data")),
        locations=locations,
        sources=sources,
        target=TargetConfig(
            enabled=bool(target_raw.get("enabled", False)),
            source=target_raw.get("source", "cepea_xls"),
            path=target_raw.get("path"),
            price_column_brl=target_raw.get("price_column_brl", "À vista R$"),
            price_column_usd=target_raw.get("price_column_usd", "À vista US$"),
            series_name=target_raw.get("series_name", "CEPEA/ESALQ - Paranaguá"),
        ),
    )
