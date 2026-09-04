from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_HEADERS = {
    "User-Agent": "pipeline-tgsi/0.1 (+https://www.eia.gov; https://portal.inmet.gov.br)",
}


class HttpRequestError(RuntimeError):
    """Raised when a remote request fails."""


def _merge_headers(headers: dict[str, str] | None) -> dict[str, str]:
    merged = dict(DEFAULT_HEADERS)
    if headers:
        merged.update(headers)
    return merged


def request_bytes(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: int = 120,
    basic_auth: tuple[str, str] | None = None,
) -> bytes:
    final_headers = _merge_headers(headers)
    if basic_auth:
        token = base64.b64encode(f"{basic_auth[0]}:{basic_auth[1]}".encode("utf-8"))
        final_headers["Authorization"] = f"Basic {token.decode('ascii')}"

    request = Request(url=url, method=method, headers=final_headers, data=data)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HttpRequestError(
            f"HTTP {exc.code} em {url}: {detail[:300]}"
        ) from exc
    except URLError as exc:
        raise HttpRequestError(f"Falha de rede em {url}: {exc.reason}") from exc


def request_text(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: int = 120,
    encoding: str = "utf-8",
    basic_auth: tuple[str, str] | None = None,
) -> str:
    return request_bytes(
        url,
        method=method,
        headers=headers,
        data=data,
        timeout=timeout,
        basic_auth=basic_auth,
    ).decode(encoding, errors="replace")


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: int = 120,
    basic_auth: tuple[str, str] | None = None,
) -> Any:
    payload = request_text(
        url,
        method=method,
        headers=headers,
        data=data,
        timeout=timeout,
        basic_auth=basic_auth,
    )
    return json.loads(payload)


def json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def add_query_params(url: str, params: dict[str, object]) -> str:
    encoded = urlencode(
        {
            key: value
            for key, value in params.items()
            if value is not None and value != ""
        },
        doseq=True,
        safe="(),'/:",
    )
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{encoded}"
