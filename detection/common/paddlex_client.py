"""HTTP compartido hacia PaddleX serving.

Cada detection/*/client.py conserva normalizadores y trackers; este módulo
centraliza ENABLE flag, base64 POST, timeout y chequeo de errorCode.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger("detection.paddlex_client")

DEFAULT_HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))


def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def is_paddlex_ok(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return data.get("errorCode") in (None, 0, "0")


def _build_predict_body(
    jpeg: bytes,
    *,
    image_field: str = "image",
    extra_json: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        image_field: base64.b64encode(jpeg).decode("ascii"),
    }
    if extra_json:
        body.update(extra_json)
    return body


def _handle_predict_payload(
    data: Any,
    *,
    log: logging.Logger,
    label: str,
    warn_on_error: bool,
) -> Optional[dict[str, Any]]:
    if not is_paddlex_ok(data):
        err = data.get("errorMsg") if isinstance(data, dict) else data
        if warn_on_error:
            log.warning("%s error: %s", label, err)
        else:
            log.debug("%s error: %s", label, err)
        return None
    return data if isinstance(data, dict) else None


async def post_image_predict(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    predict_path: str,
    jpeg: bytes,
    timeout: Optional[float] = None,
    image_field: str = "image",
    extra_json: Optional[dict[str, Any]] = None,
    log: Optional[logging.Logger] = None,
    label: str = "paddlex",
    warn_on_error: bool = False,
) -> Optional[dict[str, Any]]:
    """POST JPEG en base64 al predict path. None ante fallo de red/errorCode."""
    log = log or logger
    url = f"{base_url.rstrip('/')}{predict_path}"
    body = _build_predict_body(
        jpeg, image_field=image_field, extra_json=extra_json
    )
    try:
        resp = await client.post(
            url, json=body, timeout=timeout if timeout is not None else DEFAULT_HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        msg = "%s infer error: %s"
        if warn_on_error:
            log.warning(msg, label, exc)
        else:
            log.debug("%s infer error (isolated): %s", label, exc)
        return None

    return _handle_predict_payload(
        data, log=log, label=label, warn_on_error=warn_on_error
    )


def post_image_predict_sync(
    client: httpx.Client,
    *,
    base_url: str,
    predict_path: str,
    jpeg: bytes,
    timeout: Optional[float] = None,
    image_field: str = "image",
    extra_json: Optional[dict[str, Any]] = None,
    log: Optional[logging.Logger] = None,
    label: str = "paddlex",
    warn_on_error: bool = False,
) -> Optional[dict[str, Any]]:
    """POST sync JPEG→PaddleX (InferenceSlicer callback / harness)."""
    log = log or logger
    url = f"{base_url.rstrip('/')}{predict_path}"
    body = _build_predict_body(
        jpeg, image_field=image_field, extra_json=extra_json
    )
    try:
        resp = client.post(
            url,
            json=body,
            timeout=timeout if timeout is not None else DEFAULT_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        msg = "%s infer error: %s"
        if warn_on_error:
            log.warning(msg, label, exc)
        else:
            log.debug("%s infer error (isolated): %s", label, exc)
        return None

    return _handle_predict_payload(
        data, log=log, label=label, warn_on_error=warn_on_error
    )
