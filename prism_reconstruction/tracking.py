"""Rigid object pose from segmented 3D point tracks, with explicit failure flags."""
from __future__ import annotations

import numpy as np


def rigid_transform(source, target, weights=None):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("rigid fitting requires corresponding (N,3) point arrays, N>=3")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("rigid fitting points must be finite")
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(source),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid rigid fitting weights")
    weights = weights / weights.sum()
    a, b = weights @ source, weights @ target
    covariance = ((source-a) * weights[:, None]).T @ (target-b)
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] <= max(singular[0]*1e-8, 1e-12):
        raise ValueError("collinear/degenerate tracks cannot constrain rigid orientation")
    reflection = np.eye(3)
    reflection[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ reflection @ u.T
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, b - rotation @ a
    return transform


def track_rigid_object(points_world, visibility, reference_frame=0, threshold_m=0.03,
                       min_inliers=6, trials=128, seed=0):
    """Estimate world_from_reference poses. Failed frames remain NaN, never identity.

    `points_world` must contain only tracks selected by the object segmentation.
    Absolute metric scale must be established before specifying a metre threshold.
    """
    points, visibility = np.asarray(points_world, float), np.asarray(visibility, float)
    if points.ndim != 3 or points.shape[-1] != 3 or visibility.shape != points.shape[:2]:
        raise ValueError("expected object tracks (T,N,3), visibility (T,N)")
    if not 0 <= reference_frame < len(points) or threshold_m <= 0 or min_inliers < 3 or trials < 1:
        raise ValueError("invalid tracker configuration")
    if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
        raise ValueError("visibility must lie in [0,1]")
    reference = points[reference_frame]
    reference_valid = np.isfinite(reference).all(axis=1) & (visibility[reference_frame] >= 0.5)
    poses = np.full((len(points), 4, 4), np.nan)
    valid = np.zeros(len(points), bool)
    inlier_count = np.zeros(len(points), int)
    rmse = np.full(len(points), np.nan)
    rng = np.random.default_rng(seed)
    for t, target in enumerate(points):
        ids = np.flatnonzero(reference_valid & np.isfinite(target).all(axis=1) & (visibility[t] >= 0.5))
        if len(ids) < min_inliers:
            continue
        a, b = reference[ids], target[ids]
        best = np.zeros(len(ids), bool)
        best_error = np.inf
        for _ in range(trials):
            sample = rng.choice(len(ids), 3, replace=False)
            try:
                transform = rigid_transform(a[sample], b[sample])
            except ValueError:
                continue
            error = np.linalg.norm(a @ transform[:3, :3].T + transform[:3, 3] - b, axis=1)
            mask = error <= threshold_m
            mean_error = np.mean(error[mask]) if mask.any() else np.inf
            if mask.sum() > best.sum() or (mask.sum() == best.sum() and mean_error < best_error):
                best, best_error = mask, mean_error
        if best.sum() < max(min_inliers, int(np.ceil(0.5*len(ids)))):
            continue
        try:
            transform = rigid_transform(a[best], b[best], visibility[t, ids[best]])
        except ValueError:
            continue
        error = np.linalg.norm(a @ transform[:3, :3].T + transform[:3, 3] - b, axis=1)
        mask = error <= threshold_m
        if mask.sum() < min_inliers:
            continue
        poses[t], valid[t] = transform, True
        inlier_count[t], rmse[t] = mask.sum(), np.sqrt(np.mean(error[mask]**2))
    return dict(world_from_reference=poses, valid=valid, inlier_count=inlier_count, rmse_m=rmse)


def estimate_box_geometry(world_from_reference, object_points_world):
    """Fit a cuboid to observed object surfaces aggregated across valid frames.

    Unseen surfaces cannot be recovered from these bounds. Extents are estimates,
    not measured dimensions; callers must record that assumption in the report.
    """
    import trimesh

    canonical = []
    for pose, points in zip(world_from_reference, object_points_world):
        if not np.isfinite(pose).all():
            continue
        points = np.asarray(points, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        canonical.append((points - pose[:3, 3]) @ pose[:3, :3])
    if not canonical or sum(len(p) for p in canonical) < 20:
        raise ValueError("insufficient observed box surface points")
    points = np.concatenate(canonical)
    # Bound cost without giving any one frame additional weight.
    if len(points) > 10000:
        points = points[np.linspace(0, len(points)-1, 10000, dtype=int)]
    singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
    if singular[-1] < singular[0]*0.01:
        raise ValueError("box thickness is unobservable from near-planar points; more views or dimensions are needed")
    object_from_reference, extents = trimesh.bounds.oriented_bounds(points)
    reference_from_object = np.linalg.inv(object_from_reference)
    poses = np.asarray(world_from_reference) @ reference_from_object
    return dict(world_from_object=poses, half_extents=extents/2,
                reference_from_object=reference_from_object,
                shape_assumption="cuboid_bounds_of_observed_surfaces")
