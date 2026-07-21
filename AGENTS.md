# AGENTS.md

## Cursor Cloud specific instructions

### What this project is
Vision Intelligence (Producto B): a Docker-first computer-vision pipeline
(`Webcam/photo → MediaMTX → PaddleX inference → FastAPI adapter → AMIS/ECharts`).
The only custom code lives in `adapter.py`, `epp_core.py`, `rtsp_bridge.py` and
`rules_sink.py`; PaddleX/MediaMTX/JetLinks are third-party services. See
`README.md` for the full architecture and Docker Compose profiles.

### Dependencies / setup
Python deps (adapter + bridge + rules-sink `requirements*.txt`) are installed at
system level by the startup update script (`pip install --break-system-packages`),
so `python3` can import everything directly — no virtualenv activation needed.
Python is 3.12 here (Dockerfiles pin 3.10, but the code runs fine on 3.12).

### Tests (fast, no services needed)
Tests are stdlib `unittest`, run each file directly:
`python3 test_epp_core.py`, `python3 test_adapter_media.py`,
`python3 test_bridge_helpers.py`. There is no configured linter; use
`python3 -m py_compile adapter.py epp_core.py rtsp_bridge.py rules_sink.py`
as a basic syntax check.

### Running the app locally (dev mode, without Docker)
The `adapter` FastAPI service is the main dev-loop app (serves the dashboard +
`/ingest` + `/events` + `/media`). Run it from the repo root:
`MEDIA_DIR=/workspace/media_root python3 -m uvicorn adapter:app --host 0.0.0.0 --port 8000 --reload`
Then open `http://localhost:8000`.
Caveats:
- `MEDIA_DIR` defaults to `/media` (not writable here). Point it at a writable
  path; the `images/` subdir is created on first upload. `imagenes_muestra/` is
  gitignored and absent by default.
- `STATIC_DIR` defaults to the repo root when run from there, so leave it unset.
- The dashboard loads AMIS + ECharts from a public CDN, so the browser needs
  internet access to render.

### Exercising the pipeline without PaddleX
The full inference stack (`paddlex*`, `mediamtx`, `bridge`) runs via
`docker compose up --build` (default profile) and is heavy: Docker is NOT
installed by default in this VM, and PaddleX downloads ~2–3 GB of models on
first run. You usually do NOT need it to develop/test the custom code. To drive
consolidation locally, POST synthetic detections straight to the adapter (this
is what the bridge does), e.g.:
`curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' -d '{"detections":[{"track_id":"v-1","label":"sedan","score":0.9,"color":"white","plate":{"text":"ABC123","score":0.85},"bbox":[0,0,10,10],"track_lost":true}]}'`
then `GET /events` to see the consolidated `PerceptionEvent`. Note: uploading a
photo alone selects the media but produces no detections without the bridge.
In live/RTSP mode a track only emits after `TRACK_TTL_SECONDS` (default 10s);
send `track_lost: true` (or select a photo) to emit immediately.
