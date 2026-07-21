"""Image anomaly detection (experimental)."""

from detection.anomaly.client import (
    ENABLE_ANOMALY,
    infer_anomaly,
    normalize_anomaly_result,
)

__all__ = ["ENABLE_ANOMALY", "infer_anomaly", "normalize_anomaly_result"]
