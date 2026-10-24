"""Object-specific, simultaneous whole-body contacts with temporal hysteresis."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import box_surface, validate_poses


@dataclass(frozen=True)
class ContactConfig:
    enter_distance_m: float = 0.03
    leave_distance_m: float = 0.05
    max_penetration_m: float = 0.02
    enter_score: float = 0.5
    leave_score: float = 0.25
    min_duration_s: float = 0.06
    min_region_vertices: int = 3
    max_sample_gap_s: float = 0.15
    sticking_speed_m_s: float = 0.03


def validate_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("at least two strictly increasing finite timestamps are required")
    return times


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def infer_contacts(vertices_world, poses, half_extents, scores, regions, timestamps_s,
                   config=ContactConfig()):
    times = validate_times(timestamps_s)
    poses = validate_poses(poses)
    vertices = np.asarray(vertices_world, dtype=float)
    scores = np.asarray(scores, dtype=float)
    regions = np.asarray(regions, dtype=str)
    if vertices.ndim != 3 or vertices.shape[0] != len(times) or vertices.shape[-1] != 3:
        raise ValueError("human vertices must have shape (T,V,3), aligned to timestamps")
    if len(poses) != len(times) or scores.shape != vertices.shape[:2] or regions.shape != (vertices.shape[1],):
        raise ValueError("poses, scores, region labels and human frames must align exactly")
    if not np.isfinite(vertices).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("vertices must be finite; scores must be probabilities/binary labels in [0,1], not logits")
    if not (0 <= config.enter_distance_m <= config.leave_distance_m and config.max_penetration_m >= 0
            and 0 <= config.leave_score <= config.enter_score <= 1 and config.min_duration_s >= 0
            and config.min_region_vertices >= 1 and config.max_sample_gap_s > 0
            and config.sticking_speed_m_s >= 0):
        raise ValueError("invalid contact configuration")
    local = np.einsum("tvi,tij->tvj", vertices - poses[:, None, :3, 3], poses[:, :3, :3])
    surface, normals, signed_gap = box_surface(local, half_extents)
    admissible = signed_gap >= -config.max_penetration_m
    enter = admissible & (signed_gap <= config.enter_distance_m) & (scores >= config.enter_score)
    remain = admissible & (signed_gap <= config.leave_distance_m) & (scores >= config.leave_score)
    active = np.zeros_like(scores, dtype=bool)
    for t in range(len(times)):
        previous = active[t-1] if t and times[t] - times[t-1] <= config.max_sample_gap_s else False
        active[t] = enter[t] | (previous & remain[t])
    sample_end = np.r_[times[1:], times[-1] + np.median(np.diff(times))]
    # Never bridge unobserved time: cap each sample's support at the nominal step.
    sample_width = np.minimum(sample_end - times, np.median(np.diff(times)))
    for v in range(active.shape[1]):
        for start, end in list(runs(active[:, v])):
            split = np.r_[start, np.flatnonzero(np.diff(times[start:end]) > config.max_sample_gap_s) + start + 1, end]
            for lo, hi in zip(split[:-1], split[1:]):
                if sample_width[lo:hi].sum() + 1e-12 < config.min_duration_s:
                    active[lo:hi, v] = False
    region_names = np.unique(regions)
    counts = np.stack([active[:, regions == name].sum(axis=1) for name in region_names], axis=1)
    intervals = []
    for j, name in enumerate(region_names):
        for start, end in runs(counts[:, j] >= config.min_region_vertices):
            # Split interval reporting at timestamp discontinuities too.
            split = np.r_[start, np.flatnonzero(np.diff(times[start:end]) > config.max_sample_gap_s) + start + 1, end]
            for lo, hi in zip(split[:-1], split[1:]):
                intervals.append(dict(region=str(name), start_frame=int(lo), end_frame_exclusive=int(hi),
                                      start_time_s=float(times[lo]), last_sample_time_s=float(times[hi-1])))
    local_velocity = np.gradient(local, times, axis=0, edge_order=1)
    tangent_velocity = local_velocity - (local_velocity * normals).sum(axis=-1, keepdims=True) * normals
    speed = np.linalg.norm(tangent_velocity, axis=-1)
    mode = np.zeros(active.shape, dtype=np.uint8)
    mode[active & (speed <= config.sticking_speed_m_s)] = 1
    mode[active & (speed > config.sticking_speed_m_s)] = 2
    return dict(active=active, signed_gap_m=signed_gap.astype(np.float32),
                human_points_object=local.astype(np.float32), object_points=surface.astype(np.float32),
                outward_normals_object=normals.astype(np.float32), mode=mode,
                region_names=region_names, region_vertex_counts=counts, intervals=intervals)
