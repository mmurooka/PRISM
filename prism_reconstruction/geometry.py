"""Geometry in metres; poses map object-local column vectors into world space."""
from __future__ import annotations

import numpy as np


def validate_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("poses must be finite (T,4,4) world_from_object matrices")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5):
        raise ValueError("pose rotations must be orthonormal; apply scale to geometry separately")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
        raise ValueError("pose rotations must have determinant +1")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("invalid homogeneous pose row")
    return poses


def box_surface(points: np.ndarray, half_extents: np.ndarray):
    """Exact closest points, outward normals and signed distances to a cuboid.

    Negative distance means penetration. At an edge/corner, a deterministic
    adjacent face normal is selected; this is an approximation for force fitting.
    """
    points = np.asarray(points, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("half_extents must contain three positive finite lengths")
    if points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("surface query points must be finite (...,3)")
    q = np.abs(points) - half
    outside = np.maximum(q, 0)
    distance = np.linalg.norm(outside, axis=-1) + np.minimum(q.max(axis=-1), 0)
    closest = np.clip(points, -half, half)
    face_axis = np.argmax(q, axis=-1)
    inside = q.max(axis=-1) <= 0
    flat = closest.reshape(-1, 3)
    source = points.reshape(-1, 3)
    axes = face_axis.reshape(-1)
    rows = np.arange(len(flat))
    signs = np.where(source[rows, axes] >= 0, 1.0, -1.0)
    inside_rows = rows[inside.reshape(-1)]
    flat[inside_rows, axes[inside_rows]] = signs[inside_rows] * half[axes[inside_rows]]
    normals = np.zeros_like(source)
    normals[rows, axes] = signs
    return closest, normals.reshape(points.shape), distance


def smpl_vertex_regions(template, weights, joints, knee_radius_fraction=0.28):
    """Semantic surface regions from SMPL skinning, with explicit knee patches.

    Knee radius is a fraction of the shorter adjacent bone. This labels the
    entire knee ring; the object surface/visual prior determines its contact side.
    SMPL-X/SMPL-H weights must first be converted to SMPL topology.
    """
    template, weights, joints = map(np.asarray, (template, weights, joints))
    if template.ndim != 2 or template.shape[1] != 3 or weights.shape != (len(template), 24):
        raise ValueError("expected SMPL vertices (V,3) and skinning weights (V,24)")
    if joints.shape != (24, 3) or not all(np.isfinite(a).all() for a in (template, weights, joints)):
        raise ValueError("expected finite SMPL template, weights and 24 rest joints")
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-3):
        raise ValueError("skinning weights must be nonnegative and sum to one")
    if not 0 < knee_radius_fraction < 1:
        raise ValueError("knee_radius_fraction must lie in (0,1)")
    names = np.array([
        "pelvis", "left_thigh", "right_thigh", "torso", "left_shin", "right_shin",
        "torso", "left_foot", "right_foot", "torso", "left_foot", "right_foot",
        "neck", "left_upper_arm", "right_upper_arm", "head", "left_upper_arm",
        "right_upper_arm", "left_forearm", "right_forearm", "left_hand", "right_hand",
        "left_hand", "right_hand",
    ], dtype="U32")
    labels = names[np.argmax(weights, axis=1)]
    for side, hip, knee, ankle in [("left", 1, 4, 7), ("right", 2, 5, 8)]:
        radius = knee_radius_fraction * min(
            np.linalg.norm(joints[hip] - joints[knee]),
            np.linalg.norm(joints[ankle] - joints[knee]),
        )
        if radius <= 0:
            raise ValueError("degenerate leg bones in SMPL rest joints")
        on_leg = weights[:, [hip, knee, ankle]].sum(axis=1) > 0.5
        near_knee = np.linalg.norm(template - joints[knee], axis=1) <= radius
        labels[on_leg & near_knee] = f"{side}_knee"
    return labels
