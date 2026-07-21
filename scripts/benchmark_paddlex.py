#!/usr/bin/env python3
"""Benchmark de latencia HTTP contra servicios PaddleX del stack.

Uso (con stack levantado)::

    PYTHONPATH=. python3 scripts/benchmark_paddlex.py
    PYTHONPATH=. python3 scripts/benchmark_paddlex.py --image imagenes_muestra/foo.jpg
    PYTHONPATH=. python3 scripts/benchmark_paddlex.py --rounds 5 --json

Mide POST con JPEG (o placeholder 64x64) a cada endpoint configurado.
Documentar resultados en infra/README antes de activar VI_USE_HPIP=1.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Targets: (name, url env, default url, path env, default path)
TARGETS: list[tuple[str, str, str, str, str]] = [
    (
        "vehicles",
        "PADDLEX_URL",
        "http://127.0.0.1:8080",
        "PADDLEX_PREDICT_PATH",
        "/vehicle-attribute-recognition",
    ),
    (
        "objects",
        "PADDLEX_OBJECTS_URL",
        "http://127.0.0.1:8082",
        "PADDLEX_OBJECTS_PREDICT_PATH",
        "/object-detection",
    ),
    (
        "ocr",
        "PADDLEX_OCR_URL",
        "http://127.0.0.1:8081",
        "PADDLEX_OCR_PREDICT_PATH",
        "/ocr",
    ),
    (
        "faces",
        "PADDLEX_FACES_URL",
        "http://127.0.0.1:8083",
        "PADDLEX_FACES_PREDICT_PATH",
        "/face-detection",
    ),
    (
        "pedestrians",
        "PADDLEX_PEDESTRIANS_URL",
        "http://127.0.0.1:8084",
        "PADDLEX_PEDESTRIANS_PREDICT_PATH",
        "/pedestrian-attribute-recognition",
    ),
    (
        "scene",
        "PADDLEX_SCENE_URL",
        "http://127.0.0.1:8085",
        "PADDLEX_SCENE_PREDICT_PATH",
        "/semantic-segmentation",
    ),
    (
        "pose",
        "PADDLEX_POSE_URL",
        "http://127.0.0.1:8086",
        "PADDLEX_POSE_PREDICT_PATH",
        "/human-keypoint-detection",
    ),
    (
        "face_id",
        "PADDLEX_FACE_ID_URL",
        "http://127.0.0.1:8087",
        "PADDLEX_FACE_ID_PREDICT_PATH",
        "/face-recognition",
    ),
    (
        "signs",
        "PADDLEX_SIGNS_URL",
        "http://127.0.0.1:8088",
        "PADDLEX_SIGNS_PREDICT_PATH",
        "/object-detection",
    ),
]


def _placeholder_jpeg() -> bytes:
    """JPEG mínimo 1x1 si no hay imagen (solo mide reachability/overhead)."""
    # Minimal valid JPEG
    return base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
        "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
        "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
        "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAA"
        "AAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oA"
        "DAMBAAIQAxAAAAGf/9k="
    )


def _load_jpeg(path: Optional[str]) -> bytes:
    if not path:
        return _placeholder_jpeg()
    with open(path, "rb") as fh:
        return fh.read()


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, float]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            resp.read()
            status = int(getattr(resp, "status", 200) or 200)
    except HTTPError as exc:
        status = int(exc.code)
    except URLError as exc:
        raise ConnectionError(str(exc.reason if hasattr(exc, "reason") else exc)) from exc
    elapsed = time.perf_counter() - t0
    return status, elapsed


def bench_one(
    name: str,
    base: str,
    path: str,
    jpeg_b64: str,
    rounds: int,
    timeout: float,
) -> dict[str, Any]:
    url = f"{base.rstrip('/')}{path}"
    # OCR serving usa "file"; el resto "image"
    key = "file" if name == "ocr" or name == "text" else "image"
    payload: dict[str, Any] = {key: jpeg_b64}
    if key == "file":
        payload["fileType"] = 1

    times: list[float] = []
    last_status = 0
    error: Optional[str] = None
    for _ in range(rounds):
        try:
            status, elapsed = _post_json(url, payload, timeout)
            last_status = status
            times.append(elapsed)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            break

    if not times:
        return {
            "name": name,
            "url": url,
            "ok": False,
            "error": error or "no samples",
            "status": last_status,
        }

    return {
        "name": name,
        "url": url,
        "ok": True,
        "status": last_status,
        "rounds": len(times),
        "mean_s": round(statistics.mean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Ruta a JPEG/PNG de prueba")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--only",
        help="Comma-separated subset of target names",
    )
    args = parser.parse_args()

    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    jpeg = _load_jpeg(args.image)
    b64 = base64.b64encode(jpeg).decode("ascii")

    rows: list[dict[str, Any]] = []
    for name, url_env, url_def, path_env, path_def in TARGETS:
        if only and name not in only:
            continue
        base = os.getenv(url_env, url_def)
        path = os.getenv(path_env, path_def)
        rows.append(
            bench_one(name, base, path, b64, args.rounds, args.timeout)
        )

    if args.as_json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'name':<14} {'ok':<4} {'mean_s':>8} {'median_s':>9} {'status':>6}  url")
        for r in rows:
            if not r.get("ok"):
                print(
                    f"{r['name']:<14} {'no':<4} {'—':>8} {'—':>9} "
                    f"{r.get('status', 0):>6}  {r.get('error', '')[:60]}"
                )
            else:
                print(
                    f"{r['name']:<14} {'yes':<4} {r['mean_s']:>8.3f} "
                    f"{r['median_s']:>9.3f} {r['status']:>6}  {r['url']}"
                )
        print(
            "\nCriterio HPIP: activar VI_USE_HPIP=1 si mean_s mejora ≥~1.5× "
            "vs baseline (ver infra/README)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
