"""Pose / keypoints humanos (PaddleX human_keypoint_detection)."""

from detection.pose.client import (
    ENABLE_POSE,
    PADDLEX_POSE_URL,
    infer_pose,
    normalize_pose_result,
    reset_pose_tracker,
)

__all__ = [
    "ENABLE_POSE",
    "PADDLEX_POSE_URL",
    "infer_pose",
    "normalize_pose_result",
    "reset_pose_tracker",
]
