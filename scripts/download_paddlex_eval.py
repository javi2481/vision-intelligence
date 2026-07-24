#!/usr/bin/env python3
"""Materialize PaddleX eval fixtures under imagenes_muestra/ (host-only).

Layers: Core (default), Extended + Experimental via --packs all.
Seed=51; caps ~20/suite. FiftyOne zoo + synthetic packs. Flat root JPGs
for adapter scan; packs/ + gt/ nested alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _eval_packs import (  # noqa: E402
    CORE_PACKS,
    DEFAULT_MAX_SAMPLES,
    PACKS,
    PERSON_COCO_LABELS,
    SEED,
    SIGN_COCO_LABELS,
    VEHICLE_COCO_LABELS,
    flat_name,
    resolve_pack_names,
)

THRASH_WARN_ROOT_JPGS = 40


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dirs(out: Path) -> None:
    (out / "packs").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(parents=True, exist_ok=True)


def _copy_flat(src: Path, dest_root: Path, suite: str, index: int) -> str:
    name = flat_name(suite, index)
    dest = dest_root / name
    shutil.copy2(src, dest)
    return name


def _write_gt(out: Path, suite: str, tier: str, fixtures: list[dict[str, Any]]) -> Path:
    path = out / "gt" / f"{suite}.json"
    payload = {"suite": suite, "tier": tier, "seed": SEED, "fixtures": fixtures}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit(
            "Pillow required for synthetic packs. "
            "Install: python -m pip install -r scripts/requirements-eval.txt"
        ) from exc
    return Image, ImageDraw, ImageFont


def _materialize_synthetic_plates(
    out: Path, max_samples: int, rng: random.Random
) -> list[dict[str, Any]]:
    Image, ImageDraw, ImageFont = _pil()
    suite = "ocr_plates"
    pack_dir = out / "packs" / suite
    pack_dir.mkdir(parents=True, exist_ok=True)
    alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    fixtures: list[dict[str, Any]] = []
    for i in range(1, max_samples + 1):
        text = "".join(rng.choice(alphabet) for _ in range(6))
        img = Image.new("RGB", (320, 120), color=(20, 40, 90))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [20, 25, 300, 95], fill=(245, 245, 245), outline=(10, 10, 10), width=3
        )
        try:
            font = ImageFont.truetype("arial.ttf", 42)
        except OSError:
            font = ImageFont.load_default()
        draw.text((48, 42), text, fill=(15, 15, 15), font=font)
        pack_name = f"{i:04d}.jpg"
        pack_path = pack_dir / pack_name
        img.save(pack_path, format="JPEG", quality=92)
        flat = _copy_flat(pack_path, out, suite, i)
        fixtures.append(
            {
                "id": f"{suite}_{i:04d}",
                "file": flat,
                "pack_file": f"packs/{suite}/{pack_name}",
                "text": text,
            }
        )
    _write_gt(out, suite, PACKS[suite]["tier"], fixtures)
    return fixtures


def _materialize_synthetic_text(
    out: Path, max_samples: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Synthetic scene OCR lines (known text) — Tier A substring match."""
    Image, ImageDraw, ImageFont = _pil()
    suite = "ocr_text"
    pack_dir = out / "packs" / suite
    pack_dir.mkdir(parents=True, exist_ok=True)
    words = [
        "STOP",
        "YIELD",
        "EXIT",
        "PARKING",
        "OPEN",
        "CLOSED",
        "DANGER",
        "SCHOOL",
        "ZONE",
        "AHEAD",
    ]
    fixtures: list[dict[str, Any]] = []
    for i in range(1, max_samples + 1):
        text = rng.choice(words) + str(rng.randint(10, 99))
        img = Image.new("RGB", (400, 160), color=(240, 240, 235))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except OSError:
            font = ImageFont.load_default()
        draw.text((40, 50), text, fill=(10, 10, 10), font=font)
        pack_name = f"{i:04d}.jpg"
        pack_path = pack_dir / pack_name
        img.save(pack_path, format="JPEG", quality=90)
        flat = _copy_flat(pack_path, out, suite, i)
        fixtures.append(
            {
                "id": f"{suite}_{i:04d}",
                "file": flat,
                "pack_file": f"packs/{suite}/{pack_name}",
                "text": text,
            }
        )
    _write_gt(out, suite, PACKS[suite]["tier"], fixtures)
    return fixtures


def _materialize_synthetic_faces(
    out: Path, max_samples: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Simple oval 'faces' with known bboxes for Tier A face detection."""
    Image, ImageDraw, _Font = _pil()
    suite = "faces"
    pack_dir = out / "packs" / suite
    pack_dir.mkdir(parents=True, exist_ok=True)
    fixtures: list[dict[str, Any]] = []
    for i in range(1, max_samples + 1):
        w, h = 320, 320
        img = Image.new("RGB", (w, h), color=(60, 90, 120))
        draw = ImageDraw.Draw(img)
        # One or two faces
        n_faces = 1 + (i % 2)
        boxes: list[list[float]] = []
        labels: list[str] = []
        for k in range(n_faces):
            cx = 80 + k * 140 + rng.randint(-10, 10)
            cy = 140 + rng.randint(-20, 20)
            rw, rh = 55, 70
            x1, y1, x2, y2 = cx - rw, cy - rh, cx + rw, cy + rh
            skin = (220, 180, 150)
            draw.ellipse([x1, y1, x2, y2], fill=skin, outline=(40, 30, 20))
            draw.ellipse([cx - 15, cy - 15, cx - 5, cy - 5], fill=(20, 20, 20))
            draw.ellipse([cx + 5, cy - 15, cx + 15, cy - 5], fill=(20, 20, 20))
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append("face")
        pack_name = f"{i:04d}.jpg"
        pack_path = pack_dir / pack_name
        img.save(pack_path, format="JPEG", quality=90)
        flat = _copy_flat(pack_path, out, suite, i)
        fixtures.append(
            {
                "id": f"{suite}_{i:04d}",
                "file": flat,
                "pack_file": f"packs/{suite}/{pack_name}",
                "bboxes": boxes,
                "labels": labels,
            }
        )
    _write_gt(out, suite, PACKS[suite]["tier"], fixtures)
    return fixtures


def _materialize_synthetic_scene_cls(
    out: Path, max_samples: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Color-pattern images with class labels — Tier B schema (no hard class gate)."""
    Image, ImageDraw, _Font = _pil()
    suite = "scene_cls"
    pack_dir = out / "packs" / suite
    pack_dir.mkdir(parents=True, exist_ok=True)
    classes = [
        ("street", (80, 80, 80), (40, 120, 40)),
        ("indoor", (180, 160, 140), (100, 80, 60)),
        ("highway", (70, 70, 70), (50, 50, 120)),
        ("rural", (50, 120, 50), (100, 160, 80)),
    ]
    fixtures: list[dict[str, Any]] = []
    for i in range(1, max_samples + 1):
        label, sky, ground = classes[(i - 1) % len(classes)]
        img = Image.new("RGB", (256, 256), color=sky)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 140, 256, 256], fill=ground)
        if label == "street":
            draw.rectangle([0, 170, 256, 210], fill=(40, 40, 40))
        pack_name = f"{i:04d}.jpg"
        pack_path = pack_dir / pack_name
        img.save(pack_path, format="JPEG", quality=88)
        flat = _copy_flat(pack_path, out, suite, i)
        fixtures.append(
            {
                "id": f"{suite}_{i:04d}",
                "file": flat,
                "pack_file": f"packs/{suite}/{pack_name}",
                "label": label,
            }
        )
    _write_gt(out, suite, PACKS[suite]["tier"], fixtures)
    return fixtures


def _materialize_synthetic_anomaly(
    out: Path, max_samples: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Half normal / half anomalous — Tier C smoke (reachability), not accuracy."""
    Image, ImageDraw, _Font = _pil()
    suite = "anomaly"
    pack_dir = out / "packs" / suite
    pack_dir.mkdir(parents=True, exist_ok=True)
    fixtures: list[dict[str, Any]] = []
    for i in range(1, max_samples + 1):
        is_anom = i % 2 == 0
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        if is_anom:
            draw.rectangle([40, 40, 216, 216], fill=(220, 20, 20))
            draw.line([40, 40, 216, 216], fill=(255, 255, 0), width=8)
        else:
            draw.ellipse([80, 80, 176, 176], fill=(160, 160, 160))
        pack_name = f"{i:04d}.jpg"
        pack_path = pack_dir / pack_name
        img.save(pack_path, format="JPEG", quality=85)
        flat = _copy_flat(pack_path, out, suite, i)
        fixtures.append(
            {
                "id": f"{suite}_{i:04d}",
                "file": flat,
                "pack_file": f"packs/{suite}/{pack_name}",
                "expected": "anomaly" if is_anom else "normal",
            }
        )
    _write_gt(out, suite, PACKS[suite]["tier"], fixtures)
    return fixtures


def _materialize_synthetic_face_id(
    out: Path, max_samples: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Synthetic identities + gallery manifest for face_id Tier B."""
    Image, ImageDraw, _Font = _pil()
    suite = "face_id"
    pack_dir = out / "packs" / suite
    pack_dir.mkdir(parents=True, exist_ok=True)
    identities = ["alice", "bob", "carol", "dave"]
    gallery: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    # Gallery enrollment images (one per identity)
    for gi, ident in enumerate(identities, start=1):
        img = Image.new("RGB", (200, 200), color=(50 + gi * 20, 80, 100))
        draw = ImageDraw.Draw(img)
        skin = (
            180 + (gi * 10) % 40,
            140 + (gi * 7) % 40,
            120 + (gi * 5) % 40,
        )
        draw.ellipse([40, 30, 160, 180], fill=skin, outline=(20, 20, 20))
        gname = f"gallery_{ident}.jpg"
        gpath = pack_dir / gname
        img.save(gpath, format="JPEG", quality=90)
        gallery.append(
            {
                "identity": ident,
                "file": f"packs/{suite}/{gname}",
                "sha256": _sha256_file(gpath),
            }
        )

    for i in range(1, max_samples + 1):
        ident = identities[(i - 1) % len(identities)]
        gi = identities.index(ident) + 1
        img = Image.new("RGB", (220, 220), color=(40, 60, 90))
        draw = ImageDraw.Draw(img)
        skin = (
            180 + (gi * 10) % 40,
            140 + (gi * 7) % 40,
            120 + (gi * 5) % 40,
        )
        # Slight jitter vs gallery
        ox = rng.randint(-8, 8)
        oy = rng.randint(-8, 8)
        draw.ellipse(
            [50 + ox, 40 + oy, 170 + ox, 190 + oy], fill=skin, outline=(20, 20, 20)
        )
        pack_name = f"{i:04d}.jpg"
        pack_path = pack_dir / pack_name
        img.save(pack_path, format="JPEG", quality=90)
        flat = _copy_flat(pack_path, out, suite, i)
        fixtures.append(
            {
                "id": f"{suite}_{i:04d}",
                "file": flat,
                "pack_file": f"packs/{suite}/{pack_name}",
                "identity": ident,
                "bboxes": [[50.0 + ox, 40.0 + oy, 170.0 + ox, 190.0 + oy]],
                "labels": [ident],
            }
        )

    gallery_path = out / "gt" / "face_id_gallery.json"
    gallery_path.write_text(
        json.dumps({"seed": SEED, "suite": suite, "gallery": gallery}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_gt(out, suite, PACKS[suite]["tier"], fixtures)
    return fixtures


def _export_coco_suite(
    out: Path,
    suite: str,
    max_samples: int,
    label_filter: Optional[set[str]],
    *,
    require_boxes: bool = True,
    max_rel_area: Optional[float] = None,
) -> list[dict[str, Any]]:
    try:
        import fiftyone as fo  # noqa: F401
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise SystemExit(
            f"FiftyOne required for {suite} download. "
            "Install: python -m pip install -r scripts/requirements-eval.txt"
        ) from exc

    pack_dir = out / "packs" / suite
    pack_dir.mkdir(parents=True, exist_ok=True)

    load_cap = max(max_samples * 5, max_samples) if (label_filter or max_rel_area) else max_samples
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        max_samples=load_cap,
        shuffle=True,
        seed=SEED,
        dataset_name=f"vi-eval-{suite}-s{SEED}",
        drop_existing_dataset=True,
    )

    fixtures: list[dict[str, Any]] = []
    index = 0
    for sample in dataset:
        dets = sample["ground_truth"]
        w = float(sample.metadata.width) if sample.metadata else 0.0
        h = float(sample.metadata.height) if sample.metadata else 0.0
        boxes: list[list[float]] = []
        labels: list[str] = []
        if dets is not None and getattr(dets, "detections", None):
            for det in dets.detections:
                label = str(det.label or "").strip().lower()
                if label_filter is not None and label not in label_filter:
                    continue
                bx, by, bw, bh = det.bounding_box
                if w <= 0 or h <= 0:
                    continue
                if max_rel_area is not None and (bw * bh) > max_rel_area:
                    continue
                x1, y1 = bx * w, by * h
                x2, y2 = (bx + bw) * w, (by + bh) * h
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                labels.append(label)
        if require_boxes and not boxes:
            continue
        index += 1
        pack_name = f"{index:04d}.jpg"
        pack_path = pack_dir / pack_name
        shutil.copy2(sample.filepath, pack_path)
        flat = _copy_flat(pack_path, out, suite, index)
        fx: dict[str, Any] = {
            "id": f"{suite}_{index:04d}",
            "file": flat,
            "pack_file": f"packs/{suite}/{pack_name}",
        }
        if boxes:
            fx["bboxes"] = boxes
            fx["labels"] = labels
        fixtures.append(fx)
        if index >= max_samples:
            break

    if index < max_samples:
        print(
            f"WARNING: suite {suite!r} only materialized {index}/{max_samples} "
            f"samples after filter (seed={SEED}).",
            file=sys.stderr,
        )

    _write_gt(out, suite, PACKS[suite]["tier"], fixtures)
    try:
        dataset.delete()
    except Exception:  # noqa: BLE001
        pass
    return fixtures


def _write_manifest(
    out: Path,
    packs_arg: str,
    suite_fixtures: dict[str, list[dict[str, Any]]],
) -> Path:
    suites: dict[str, Any] = {}
    for suite, fixtures in suite_fixtures.items():
        images = []
        for fx in fixtures:
            flat_path = out / fx["file"]
            images.append(
                {
                    "id": fx["id"],
                    "file": fx["file"],
                    "sha256": _sha256_file(flat_path),
                    "gt": f"gt/{suite}.json#{fx['id']}",
                }
            )
        suites[suite] = {"images": images, "count": len(images)}
    payload = {
        "seed": SEED,
        "packs": packs_arg,
        "suites": suites,
    }
    path = out / "gt" / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def download_packs(
    out: Path,
    pack_names: list[str],
    packs_arg: str,
    max_samples: Optional[int],
) -> dict[str, list[dict[str, Any]]]:
    _ensure_dirs(out)
    rng = random.Random(SEED)
    suite_fixtures: dict[str, list[dict[str, Any]]] = {}
    root_writes = 0

    for name in pack_names:
        meta = PACKS.get(name)
        if meta is None:
            raise SystemExit(f"Unknown pack: {name}")
        if meta["source"] == "stub":
            print(f"SKIP stub pack {name!r}.", file=sys.stderr)
            continue
        cap = int(
            max_samples if max_samples is not None else meta.get("max_samples", DEFAULT_MAX_SAMPLES)
        )
        if max_samples is None:
            cap = min(cap, DEFAULT_MAX_SAMPLES)
        print(f"==> materializing {name} (source={meta['source']}, max={cap}, seed={SEED})")

        src = meta["source"]
        if src == "synthetic_plates":
            fixtures = _materialize_synthetic_plates(out, cap, rng)
        elif src == "synthetic_text":
            fixtures = _materialize_synthetic_text(out, cap, rng)
        elif src == "synthetic_faces":
            fixtures = _materialize_synthetic_faces(out, cap, rng)
        elif src == "synthetic_scene_cls":
            fixtures = _materialize_synthetic_scene_cls(out, cap, rng)
        elif src == "synthetic_anomaly":
            fixtures = _materialize_synthetic_anomaly(out, cap, rng)
        elif src == "synthetic_face_id":
            fixtures = _materialize_synthetic_face_id(out, cap, rng)
        elif src in ("fiftyone_coco", "fiftyone_coco_small"):
            filt = set(meta["coco_filter"]) if meta.get("coco_filter") else None
            if name == "vehicles" and filt is None:
                filt = set(VEHICLE_COCO_LABELS)
            if name == "signs" and filt is None:
                filt = set(SIGN_COCO_LABELS)
            if name in ("pose", "pedestrians") and filt is None:
                filt = set(PERSON_COCO_LABELS)
            max_rel = meta.get("max_rel_area") if src == "fiftyone_coco_small" else None
            fixtures = _export_coco_suite(
                out,
                name,
                cap,
                filt,
                require_boxes=bool(meta.get("require_boxes", True)),
                max_rel_area=float(max_rel) if max_rel is not None else None,
            )
        else:
            raise SystemExit(f"Unsupported source for {name}: {src}")

        suite_fixtures[name] = fixtures
        root_writes += len(fixtures)

    _write_manifest(out, packs_arg, suite_fixtures)

    if root_writes >= THRASH_WARN_ROOT_JPGS:
        print(
            f"WARNING: wrote {root_writes} flat JPGs under {out}. "
            "Media-watch thrash risk — pause adapter watch or clear imagenes_muestra "
            "root between bulk runs (see imagenes_muestra/README.md).",
            file=sys.stderr,
        )
    else:
        print(
            f"Wrote {root_writes} flat root JPGs (cap guidance ~{DEFAULT_MAX_SAMPLES}/suite)."
        )
    return suite_fixtures


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packs",
        default="core",
        choices=("core", "all"),
        help="Pack layer set (default: core)",
    )
    parser.add_argument(
        "--out",
        default="imagenes_muestra",
        help="Output root (default: imagenes_muestra)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=f"Per-suite cap (default: {DEFAULT_MAX_SAMPLES})",
    )
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        names = resolve_pack_names(args.packs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.packs == "core":
        assert set(names) == set(CORE_PACKS)

    download_packs(out, names, args.packs, args.max_samples)
    print(f"Done. Manifest: {out / 'gt' / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
