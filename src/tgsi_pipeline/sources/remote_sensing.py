from __future__ import annotations

import csv
import io
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from ..http import HttpRequestError, json_body, request_json, request_text
from ..models import DateRange, Location, RemoteSensingConfig
from ..utils import normalize_text, parse_date_any, parse_float


APPEEARS_API_BASE = "https://appeears.earthdatacloud.nasa.gov/api"


@dataclass(slots=True)
class LayerSpec:
    product_id: str
    layer_name: str
    value_field: str
    source_field: str
    scale_factor: float | None
    add_offset: float | None
    fill_value: float | None
    valid_min: float | None
    valid_max: float | None

    @property
    def source_value(self) -> str:
        return f"AppEEARS:{self.product_id}:{self.layer_name}"


def fetch_remote_sensing(
    location: Location,
    date_range: DateRange,
    config: RemoteSensingConfig,
) -> dict[str, list[dict[str, Any]]]:
    if not config.enabled:
        return {"ndvi": [], "soil_moisture": []}

    username = os.getenv(config.earthdata_username_env, "").strip()
    password = os.getenv(config.earthdata_password_env, "").strip()
    if not username or not password:
        raise RuntimeError(
            "Sensoriamento remoto habilitado, mas as credenciais Earthdata nao foram encontradas."
        )

    token = _login(username, password)
    try:
        ndvi_specs = _resolve_layer_specs(
            config.ndvi_product_id,
            requested_layers=config.ndvi_layer,
            variable="ndvi",
        )
        soil_specs = _resolve_layer_specs(
            config.soil_moisture_product_id,
            requested_layers=config.soil_moisture_layer,
            variable="soil_moisture",
        )

        ndvi_rows = _run_point_extraction(
            token,
            location,
            date_range,
            specs=ndvi_specs,
            task_name=f"{location.id}-ndvi-{int(time.time())}",
            value_field="ndvi",
            source_field="ndvi_source",
            timeout_seconds=config.task_timeout_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        soil_rows = _run_point_extraction(
            token,
            location,
            date_range,
            specs=soil_specs,
            task_name=f"{location.id}-soil-moisture-{int(time.time())}",
            value_field="soil_moisture_m3m3",
            source_field="soil_moisture_source",
            timeout_seconds=config.task_timeout_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        return {
            "ndvi": sorted(ndvi_rows, key=lambda item: item["date"]),
            "soil_moisture": sorted(soil_rows, key=lambda item: item["date"]),
        }
    finally:
        _logout(token)


def _login(username: str, password: str) -> str:
    payload = request_json(
        f"{APPEEARS_API_BASE}/login",
        method="POST",
        data=b"",
        basic_auth=(username, password),
    )
    token = payload.get("token") if isinstance(payload, dict) else None
    if not token:
        raise RuntimeError("Nao foi possivel autenticar no AppEEARS.")
    return str(token)


def _logout(token: str) -> None:
    try:
        request_json(
            f"{APPEEARS_API_BASE}/logout",
            method="POST",
            data=b"",
            headers={"Authorization": f"Bearer {token}"},
        )
    except Exception:  # noqa: BLE001
        return


def _resolve_layer_specs(
    product_id: str,
    *,
    requested_layers: str | None,
    variable: str,
) -> list[LayerSpec]:
    product = request_json(f"{APPEEARS_API_BASE}/product/{product_id}")
    if not isinstance(product, dict) or not product:
        raise RuntimeError(f"Produto AppEEARS vazio ou invalido: {product_id}")

    layer_names = _choose_layer_names(product, requested_layers=requested_layers, variable=variable)
    if not layer_names:
        raise RuntimeError(f"Nenhuma layer elegivel encontrada para {product_id}.")

    specs: list[LayerSpec] = []
    for layer_name in layer_names:
        metadata = product.get(layer_name, {})
        specs.append(
            LayerSpec(
                product_id=product_id,
                layer_name=layer_name,
                value_field="ndvi" if variable == "ndvi" else "soil_moisture_m3m3",
                source_field="ndvi_source" if variable == "ndvi" else "soil_moisture_source",
                scale_factor=parse_float(metadata.get("ScaleFactor")),
                add_offset=parse_float(metadata.get("AddOffset")),
                fill_value=parse_float(metadata.get("FillValue")),
                valid_min=parse_float(metadata.get("ValidMin")),
                valid_max=parse_float(metadata.get("ValidMax")),
            )
        )
    return specs


def _choose_layer_names(
    product: dict[str, Any],
    *,
    requested_layers: str | None,
    variable: str,
) -> list[str]:
    if requested_layers:
        output: list[str] = []
        for layer_name in [item.strip() for item in requested_layers.split(",") if item.strip()]:
            if layer_name not in product:
                raise RuntimeError(f"Layer {layer_name!r} nao encontrada no produto.")
            output.append(layer_name)
        return output

    if variable == "ndvi":
        chosen = _pick_ndvi_layers(product)
    elif variable == "soil_moisture":
        chosen = _pick_soil_moisture_layers(product)
    else:
        raise ValueError(f"Variavel remota invalida: {variable}")

    return chosen


def _pick_ndvi_layers(product: dict[str, Any]) -> list[str]:
    priority = ["NDVI", "1 km 16 days NDVI", "_1_km_16_days_NDVI", "_250m_16_days_NDVI"]
    for candidate in priority:
        if candidate in product and not _is_qa_layer(candidate, product[candidate]):
            return [candidate]

    matches = [
        layer_name
        for layer_name, metadata in product.items()
        if "ndvi" in _normalize_key(layer_name) and not _is_qa_layer(layer_name, metadata)
    ]
    if not matches:
        return []
    matches.sort(key=lambda item: (_normalize_key(item) != "ndvi", len(item)))
    return [matches[0]]


def _pick_soil_moisture_layers(product: dict[str, Any]) -> list[str]:
    preferred = [
        "soil_moisture_am",
        "soil_moisture_pm",
        "soil_moisture",
        "soil_moisture_9km_am",
        "soil_moisture_9km_pm",
    ]
    chosen = [
        layer_name
        for layer_name in preferred
        if layer_name in product and not _is_qa_layer(layer_name, product[layer_name])
    ]
    if chosen:
        return chosen[:2]

    normalized_items = [
        (layer_name, _normalize_key(layer_name), metadata)
        for layer_name, metadata in product.items()
    ]

    am_candidates = [
        layer_name
        for layer_name, norm, metadata in normalized_items
        if "soilmoisture" in norm and norm.endswith("soilmoistuream") and not _is_qa_layer(layer_name, metadata)
    ]
    pm_candidates = [
        layer_name
        for layer_name, norm, metadata in normalized_items
        if "soilmoisture" in norm and norm.endswith("soilmoisturepm") and not _is_qa_layer(layer_name, metadata)
    ]
    chosen = []
    if am_candidates:
        chosen.append(sorted(am_candidates, key=len)[0])
    if pm_candidates:
        chosen.append(sorted(pm_candidates, key=len)[0])
    if chosen:
        return chosen[:2]

    matches = [
        layer_name
        for layer_name, norm, metadata in normalized_items
        if "soilmoisture" in norm and not _is_qa_layer(layer_name, metadata)
    ]
    matches.sort(key=len)
    return matches[:2]


def _is_qa_layer(layer_name: str, metadata: Any) -> bool:
    probe = _normalize_key(layer_name)
    if any(token in probe for token in ("qa", "quality", "qc", "flag")):
        return True
    if isinstance(metadata, dict):
        explicit_flag = metadata.get("IsQA")
        if isinstance(explicit_flag, bool):
            return explicit_flag
        description = " ".join(
            str(metadata.get(key, ""))
            for key in ("Description", "Layer", "Units", "Group")
        )
        description_probe = _normalize_key(description)
        if any(token in description_probe for token in ("quality", "qa", "flag", "bitmask")):
            return True
    return False


def _run_point_extraction(
    token: str,
    location: Location,
    date_range: DateRange,
    *,
    specs: list[LayerSpec],
    task_name: str,
    value_field: str,
    source_field: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> list[dict[str, Any]]:
    task_id = _submit_point_task(
        token,
        location,
        date_range,
        task_name=task_name,
        specs=specs,
    )
    _wait_for_task(
        token,
        task_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    csv_text = _download_results_csv(
        token,
        task_id,
        specs[0].product_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    return _parse_point_csv(
        csv_text,
        location=location,
        date_range=date_range,
        specs=specs,
        value_field=value_field,
        source_field=source_field,
    )


def _submit_point_task(
    token: str,
    location: Location,
    date_range: DateRange,
    *,
    task_name: str,
    specs: list[LayerSpec],
) -> str:
    payload = {
        "task_type": "point",
        "task_name": task_name,
        "params": {
            "dates": [
                {
                    "startDate": date_range.start.strftime("%m-%d-%Y"),
                    "endDate": date_range.end.strftime("%m-%d-%Y"),
                }
            ],
            "layers": [
                {"product": spec.product_id, "layer": spec.layer_name}
                for spec in specs
            ],
            "coordinates": [
                {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                }
            ],
        },
    }
    response = request_json(
        f"{APPEEARS_API_BASE}/task",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json_body(payload),
        timeout=240,
    )
    if isinstance(response, dict):
        task_id = response.get("task_id") or response.get("taskid") or response.get("task")
        if task_id:
            return str(task_id)
    raise RuntimeError(f"Resposta inesperada ao criar task AppEEARS: {response!r}")


def _wait_for_task(
    token: str,
    task_id: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    started = time.time()
    last_status = "submitted"
    while True:
        if time.time() - started > timeout_seconds:
            raise TimeoutError(
                f"Timeout aguardando task {task_id} do AppEEARS. Ultimo status: {last_status}"
            )

        status_payload = request_json(
            f"{APPEEARS_API_BASE}/status/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=240,
        )
        status = _extract_task_status(status_payload)
        last_status = status
        normalized = status.strip().lower()
        if normalized in {"done", "complete", "completed", "success", "successful"}:
            return
        if normalized in {"error", "failed", "failure", "deleted"}:
            raise RuntimeError(f"Task {task_id} retornou status {status!r}.")
        time.sleep(poll_interval_seconds)


def _extract_task_status(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("status", "state", "task_status"):
            value = payload.get(key)
            if value:
                return str(value)
        return str(payload)
    if isinstance(payload, list) and payload:
        item = payload[0]
        if isinstance(item, dict):
            for key in ("status", "state", "task_status"):
                value = item.get(key)
                if value:
                    return str(value)
        return str(item)
    return str(payload)


def _download_results_csv(
    token: str,
    task_id: str,
    product_id: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> str:
    started = time.time()
    while True:
        try:
            bundle_payload = request_json(
                f"{APPEEARS_API_BASE}/bundle/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=240,
            )
            bundle_files = _extract_bundle_files(bundle_payload, task_id)
            file_id = _pick_bundle_file_id(bundle_files, product_id)
            return request_text(
                f"{APPEEARS_API_BASE}/bundle/{task_id}/{file_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=240,
            )
        except HttpRequestError as exc:
            if time.time() - started > timeout_seconds:
                raise RuntimeError(
                    f"Timeout aguardando o bundle final do AppEEARS para {product_id}: {exc}"
                ) from exc
            time.sleep(poll_interval_seconds)


def _extract_bundle_files(bundle_payload: Any, task_id: str) -> list[dict[str, Any]]:
    if isinstance(bundle_payload, list):
        return [item for item in bundle_payload if isinstance(item, dict)]
    if isinstance(bundle_payload, dict):
        files = bundle_payload.get("files")
        if isinstance(files, list):
            return [item for item in files if isinstance(item, dict)]
    raise RuntimeError(f"Bundle AppEEARS invalido para task {task_id}: {bundle_payload!r}")


def _pick_bundle_file_id(bundle: list[dict[str, Any]], product_id: str) -> str:
    normalized_product = _normalize_key(product_id)
    candidates: list[tuple[int, str]] = []
    for item in bundle:
        file_id = item.get("file_id")
        file_name = str(item.get("file_name", ""))
        file_type = str(item.get("file_type", "")).lower()
        if not file_id:
            continue
        if file_type and file_type != "csv":
            continue
        probe = _normalize_key(file_name)
        score = 0
        if probe.endswith("resultscsv"):
            score += 2
        if normalized_product in probe:
            score += 3
        if "csv" in probe:
            score += 1
        candidates.append((score, str(file_id)))

    if not candidates:
        raise RuntimeError(f"Nenhum arquivo CSV encontrado no bundle do AppEEARS para {product_id}.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _parse_point_csv(
    csv_text: str,
    *,
    location: Location,
    date_range: DateRange,
    specs: list[LayerSpec],
    value_field: str,
    source_field: str,
) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: dict[str, dict[str, Any]] = {}
    for record in reader:
        current_date = _extract_record_date(record)
        if current_date is None:
            continue
        if not (date_range.start <= current_date <= date_range.end):
            continue

        values = [_extract_layer_value(record, spec) for spec in specs]
        usable_values = [value for value in values if value is not None]
        if not usable_values:
            continue

        output_date = current_date.isoformat()
        rows[output_date] = {
            "date": output_date,
            "location_id": location.id,
            "location_name": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude,
            value_field: round(sum(usable_values) / len(usable_values), 6),
            source_field: ",".join(spec.source_value for spec, value in zip(specs, values) if value is not None),
        }
    return [rows[key] for key in sorted(rows)]


def _extract_record_date(record: dict[str, str]) -> date | None:
    preferred_keys = []
    for key in record:
        probe = _normalize_key(key)
        if probe in {"date", "calendardate"}:
            preferred_keys.append(key)
        elif probe.endswith("date") and "start" not in probe and "end" not in probe:
            preferred_keys.append(key)

    for key in preferred_keys:
        value = (record.get(key) or "").strip()
        if not value:
            continue
        try:
            return parse_date_any(value)
        except ValueError:
            continue

    year_key = next((key for key in record if _normalize_key(key) == "year"), None)
    doy_key = next(
        (key for key in record if _normalize_key(key) in {"doy", "dayofyear"}),
        None,
    )
    if year_key and doy_key:
        year = parse_float(record.get(year_key))
        day_of_year = parse_float(record.get(doy_key))
        if year is not None and day_of_year is not None:
            return date(int(year), 1, 1) + timedelta(days=int(day_of_year) - 1)

    return None


def _extract_layer_value(record: dict[str, str], spec: LayerSpec) -> float | None:
    column_name = _find_value_column(record, spec)
    if column_name is None:
        return None
    raw_value = parse_float(record.get(column_name))
    if raw_value is None:
        return None

    if spec.fill_value is not None and raw_value == spec.fill_value:
        return None
    if spec.valid_min is not None and raw_value < spec.valid_min:
        return None
    if spec.valid_max is not None and raw_value > spec.valid_max and not _needs_scaling(raw_value, spec):
        return None

    value = raw_value
    if _needs_scaling(raw_value, spec):
        if spec.scale_factor is not None:
            value *= spec.scale_factor
        if spec.add_offset is not None:
            value += spec.add_offset

    return value


def _find_value_column(record: dict[str, str], spec: LayerSpec) -> str | None:
    normalized_record = {_normalize_key(key): key for key in record}
    candidates = [
        spec.layer_name,
        f"{spec.product_id}_{spec.layer_name}",
        f"{spec.product_id.replace('.', '_')}_{spec.layer_name}",
        f"{spec.product_id.replace('.', '-')}_{spec.layer_name}",
        f"{spec.product_id.split('.')[0]}_{spec.layer_name}",
        spec.layer_name.replace(" ", "_"),
    ]

    candidate_keys = {_normalize_key(item) for item in candidates}
    for normalized, original in normalized_record.items():
        if normalized in candidate_keys:
            return original

    layer_key = _normalize_key(spec.layer_name)
    for normalized, original in normalized_record.items():
        if normalized.endswith(layer_key):
            return original
    return None


def _needs_scaling(raw_value: float, spec: LayerSpec) -> bool:
    if spec.scale_factor is None:
        return False
    if spec.scale_factor == 1 and (spec.add_offset is None or spec.add_offset == 0):
        return False

    layer_probe = _normalize_key(spec.layer_name)
    if "ndvi" in layer_probe and abs(raw_value) > 1.2:
        return True
    if "soilmoisture" in layer_probe and raw_value > 1.0:
        return True
    if spec.valid_max is not None and raw_value > spec.valid_max:
        return True
    return False


def _normalize_key(value: str) -> str:
    return "".join(character for character in normalize_text(value) if character.isalnum())
