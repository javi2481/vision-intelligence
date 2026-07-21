"""Tracker mínimo por IoU para asignar track_id estables entre detecciones.

No es MOT de modelo: solo orquestación local para que el adapter consolide
por track_id. Usado por vehicles (prefijo v-) y objects (prefijo o-).
"""

from __future__ import annotations


def iou(a: list[float], b: list[float]) -> float:
    """IoU entre dos bboxes [x1,y1,x2,y2]. 0.0 si no hay intersección."""
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


class IoUTracker:
    """Asigna track_id reutilizando el bbox previo con mayor IoU sobre umbral."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: dict[str, list[float]] = {}

    def assign(self, boxes: list[list[float]]) -> list[str]:
        """Devuelve un track_id (str) por cada bbox de entrada, en el mismo orden."""
        used: set[str] = set()
        ids: list[str] = []
        for bbox in boxes:
            best_id, best_iou = None, 0.0
            for tid, prev in self._tracks.items():
                if tid in used:
                    continue
                score = iou(bbox, prev)
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
        self._tracks = {tid: self._tracks[tid] for tid in used}
        return ids
