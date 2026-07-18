"""
Puente resiliente: RTSP (MediaMTX) → PaddleX → Adapter.

PaddleX 3.x: pipeline `vehicle_attribute_recognition`
  POST /vehicle-attribute-recognition  { "image": "<base64>" }

PP-Vehicle (PaddleDetection) no existe en PaddleX 3. Este pipeline detecta
vehículos + atributos (color/tipo). No trae track_id ni patente OCR:
asignamos track_id con IoU tracker mínimo (orquestación, no modelo propio).
Patente queda null hasta cablear OCR opcional.

Resiliencia: backoff RTSP, degradación si PaddleX cae, DEMO_MODE sin cámara.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import httpx

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] bridge: %(message)s",
)
logger = logging.getLogger("rtsp_bridge")

RTSP_URL = os.getenv("RTSP_URL", "rtsp://mediamtx:8554/webcam")
ADAPTER_INGEST_URL = os.getenv(
    "ADAPTER_INGEST_URL", "http://adapter:8000/ingest"
)
PADDLEX_URL = os.getenv("PADDLEX_URL", "http://paddlex:8080")
PADDLEX_PREDICT_PATH = os.getenv(
    "PADDLEX_PREDICT_PATH", "/vehicle-attribute-recognition"
)
FPS = float(os.getenv("BRIDGE_FPS", "2"))
FRAME_INTERVAL = 1.0 / max(FPS, 0.1)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
BACKOFF_INITIAL = float(os.getenv("BACKOFF_INITIAL", "1.0"))
BACKOFF_MAX = float(os.getenv("BACKOFF_MAX", "30.0"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))


class IoUTracker:
    """Tracker mínimo por IoU para asignar track_id estables entre frames."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: dict[str, list[float]] = {}  # id -> bbox

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def assign(self, boxes: list[list[float]]) -> list[str]:
        used: set[str] = set()
        ids: list[str] = []
        for bbox in boxes:
            best_id, best_iou = None, 0.0
            for tid, prev in self._tracks.items():
                if tid in used:
                    continue
                score = self._iou(bbox, prev)
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None and best_iou >= self.iou_threshold:
                self._tracks[best_id] = bbox
                used.add(best_id)
                ids.append(best_id)
            else:
                tid = str(self._next_id)
                self._next_id += 1
                self._tracks[tid] = bbox
                used.add(tid)
                ids.append(tid)
        # Limpiar tracks no vistos este frame
        self._tracks = {tid: self._tracks[tid] for tid in used}
        return ids


_tracker = IoUTracker(IOU_THRESHOLD)


def _open_capture(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _encode_jpeg(frame) -> Optional[bytes]:
    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return None
    return buf.tobytes()


def _parse_attr_labels(labels: list[Any], scores: list[Any]) -> tuple[Optional[str], Optional[str]]:
    """Extrae color y vehicle_type de labels tipo 'red(红色)', 'sedan(轿车)'."""
    color, vtype = None, None
    color_keys = {
        "red", "blue", "green", "yellow", "white", "black", "brown",
        "silver", "grey", "gray", "orange", "purple", "gold",
    }
    type_keys = {
        "sedan", "suv", "van", "truck", "bus", "mpv", "pickup", "car",
    }
    for label, score in zip(labels or [], scores or []):
        raw = str(label).split("(")[0].strip().lower()
        if raw in color_keys and color is None:
            color = raw
        elif raw in type_keys and vtype is None:
            vtype = raw
        elif color is None and raw not in type_keys:
            # primer atributo no-tipo → color candidato
            color = raw
    return color, vtype


def _normalize_paddlex_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Traduce respuesta vehicle_attribute_recognition → dicts del adaptador.

    Serving API PaddleX 3.7 (HTTP):
      { "result": { "vehicles": [ { "bbox", "attributes", "score" } ], "image": "..." } }

    Predict local (docs):
      { "result": { "boxes": [ { labels, cls_scores, det_score, coordinate } ] } }
    """
    result = data.get("result", data)
    boxes: list[dict[str, Any]] = []
    if isinstance(result, dict):
        raw = (
            result.get("vehicles")
            or result.get("boxes")
            or result.get("detections")
            or []
        )
        if isinstance(raw, list):
            boxes = raw
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and (
                "boxes" in item or "vehicles" in item
            ):
                boxes.extend(item.get("boxes") or item.get("vehicles") or [])
            elif isinstance(item, dict) and (
                "coordinate" in item or "bbox" in item
            ):
                boxes.append(item)

    coords: list[list[float]] = []
    meta: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]

        # Formato serving: attributes = [{label, score}, ...] o labels planos
        labels: list[Any] = []
        scores: list[Any] = []
        attrs = box.get("attributes")
        if isinstance(attrs, list):
            for a in attrs:
                if isinstance(a, dict):
                    labels.append(a.get("label") or a.get("name") or "")
                    scores.append(a.get("score") or a.get("confidence") or 0.0)
                else:
                    labels.append(a)
                    scores.append(1.0)
        else:
            labels = list(box.get("labels") or [])
            scores = list(box.get("cls_scores") or [])

        color, vtype = _parse_attr_labels(labels, scores)
        coords.append(bbox)
        meta.append(
            {
                "label": vtype or "vehicle",
                "score": float(
                    box.get("score")
                    or box.get("det_score")
                    or 0.0
                ),
                "color": color,
                "bbox": bbox,
            }
        )

    track_ids = _tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    detections: list[dict[str, Any]] = []
    for tid, m in zip(track_ids, meta):
        detections.append(
            {
                "track_id": tid,
                "label": m["label"],
                "score": m["score"],
                "color": m["color"],
                "bbox": m["bbox"],
                "plate": None,  # OCR patente: fase siguiente
                "frame_ts": now,
            }
        )
    return detections


def _demo_detections() -> list[dict[str, Any]]:
    track_id = str(random.randint(1, 5))
    plates = ["ABC123", "ABC123", "ABG123", "XYZ789", "ABC123"]
    colors = ["white", "white", "silver", "black", "white"]
    return [
        {
            "track_id": track_id,
            "label": random.choice(["car", "truck", "bus"]),
            "score": round(random.uniform(0.6, 0.98), 3),
            "color": random.choice(colors),
            "bbox": [100, 120, 340, 280],
            "plate": {
                "text": random.choice(plates),
                "score": round(random.uniform(0.5, 0.97), 3),
            },
            "frame_ts": datetime.now(timezone.utc).isoformat(),
        }
    ]


async def _post_json(client: httpx.AsyncClient, url: str, payload: Any) -> bool:
    try:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("POST %s failed: %s", url, exc)
        return False


async def _infer_paddlex(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST base64 a /vehicle-attribute-recognition y normaliza a detecciones."""
    url = f"{PADDLEX_URL.rstrip('/')}{PADDLEX_PREDICT_PATH}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("PaddleX infer error: %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    if data.get("errorCode") not in (None, 0, "0"):
        logger.warning("PaddleX error: %s", data.get("errorMsg"))
        return None
    return _normalize_paddlex_result(data)


async def _notify_degraded(client: httpx.AsyncClient) -> None:
    await _post_json(client, ADAPTER_INGEST_URL, {"degraded": True})


async def run_loop() -> None:
    backoff = BACKOFF_INITIAL
    logger.info(
        "Bridge start rtsp=%s paddlex=%s%s adapter=%s demo=%s fps=%.1f",
        RTSP_URL,
        PADDLEX_URL,
        PADDLEX_PREDICT_PATH,
        ADAPTER_INGEST_URL,
        DEMO_MODE,
        FPS,
    )

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while True:
            cap: Optional[cv2.VideoCapture] = None
            try:
                if DEMO_MODE:
                    detections = _demo_detections()
                    await _post_json(
                        client, ADAPTER_INGEST_URL, {"detections": detections}
                    )
                    await asyncio.sleep(FRAME_INTERVAL)
                    backoff = BACKOFF_INITIAL
                    continue

                cap = _open_capture(RTSP_URL)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open RTSP: {RTSP_URL}")

                logger.info("RTSP connected: %s", RTSP_URL)
                backoff = BACKOFF_INITIAL
                last_sent = 0.0

                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("RTSP read failed")

                    now = time.monotonic()
                    if now - last_sent < FRAME_INTERVAL:
                        await asyncio.sleep(0.01)
                        continue
                    last_sent = now

                    jpeg = _encode_jpeg(frame)
                    if jpeg is None:
                        continue

                    detections = await _infer_paddlex(client, jpeg)
                    if detections is None:
                        await _notify_degraded(client)
                        continue

                    await _post_json(
                        client,
                        ADAPTER_INGEST_URL,
                        {"detections": detections},
                    )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Bridge error: %s — reconnect in %.1fs", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, BACKOFF_MAX)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass


def main() -> None:
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        logger.info("Bridge stopped")


if __name__ == "__main__":
    main()
