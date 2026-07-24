# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# OpenCV headless + libs mínimas para imread/overlay (sin ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY bridge/requirements.txt /app/requirements-bridge.txt
ENV PIP_REQUIRE_HASHES=0
# supervision pulls opencv-python (GUI); keep only headless for the bridge image.
RUN pip install --no-cache-dir --retries 10 -r /app/requirements-bridge.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84 \
    && python -c "import importlib.metadata as md; import cv2, supervision as sv; names={d.metadata['Name'].lower() for d in md.distributions()}; assert 'opencv-python-headless' in names and 'opencv-python' not in names; print('cv2', cv2.__version__, 'sv', sv.__version__)"

COPY detection/ /app/detection/
COPY bridge/ /app/bridge/

CMD ["python", "-m", "bridge.main"]
