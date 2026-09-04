from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(slots=True)
class DateRange:
    start: date
    end: date

    def years(self) -> range:
        return range(self.start.year, self.end.year + 1)


@dataclass(slots=True)
class Location:
    id: str
    name: str
    latitude: float
    longitude: float
    inmet_station_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClimateConfig:
    enabled: bool = True
    nasa_power_fallback: bool = True
    nasa_power_parameters: list[str] = field(
        default_factory=lambda: [
            "PRECTOTCORR",
            "T2M",
            "T2M_MAX",
            "T2M_MIN",
            "RH2M",
            "WS2M",
        ]
    )


@dataclass(slots=True)
class RemoteSensingConfig:
    enabled: bool = False
    earthdata_username_env: str = "EARTHDATA_USERNAME"
    earthdata_password_env: str = "EARTHDATA_PASSWORD"
    ndvi_product_id: str = "MOD13Q1.061"
    ndvi_layer: str | None = None
    soil_moisture_product_id: str = "SPL3SMP_E.006"
    soil_moisture_layer: str | None = None
    poll_interval_seconds: int = 15
    task_timeout_seconds: int = 1800


@dataclass(slots=True)
class OilConfig:
    provider: str = "eia"
    frequency: str = "daily"


@dataclass(slots=True)
class EconomicsConfig:
    enabled: bool = True
    fx_currency: str = "USD"
    oil: OilConfig = field(default_factory=OilConfig)


@dataclass(slots=True)
class SourcesConfig:
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    remote_sensing: RemoteSensingConfig = field(default_factory=RemoteSensingConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)


@dataclass(slots=True)
class TargetConfig:
    enabled: bool = False
    source: str = "cepea_xls"
    path: str | None = None
    price_column_brl: str = "À vista R$"
    price_column_usd: str = "À vista US$"
    series_name: str = "CEPEA/ESALQ - Paranaguá"


@dataclass(slots=True)
class ProjectConfig:
    date_range: DateRange
    target_frequencies: list[str]
    output_base_dir: Path
    locations: list[Location]
    sources: SourcesConfig
    target: TargetConfig = field(default_factory=TargetConfig)
