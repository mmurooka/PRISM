"""Inverse dynamics evidence for contact, including gravity and scene support.

Unknown mass is normalized to 1 kg: force values then mean N/kg of object mass.
This estimates explanatory forces, not force measurements or unique contacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .contacts import validate_times
from .geometry import validate_poses


@dataclass(frozen=True)
class ForceCandidate:
    region: str
    point_world: np.ndarray
    normal_world: np.ndarray  # direction of compressive force ON the object
    prior: float = 1.0
    friction: float = 0.6
    source: str = "human"


def local_polynomial(times, values, window_s=0.20, derivative=0):
    """Time-aware quadratic fit; derivatives amplify noise without smoothing."""
    times = validate_times(times)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(times), 3) or not np.isfinite(values).all() or len(times) < 3:
        raise ValueError("motion derivatives require at least three finite 3D samples")
    result = np.empty_like(values)
    for i, t in enumerate(times):
        ids = np.flatnonzero(np.abs(times - t) <= window_s / 2)
        if len(ids) < 3:
            ids = np.sort(np.argsort(np.abs(times - t))[:3])
        dt = times[ids] - t
        design = np.stack([np.ones_like(dt), dt, 0.5 * dt**2], axis=1)
        coefficients = np.linalg.lstsq(design, values[ids], rcond=None)[0]
        result[i] = coefficients[derivative]
    return result


def required_wrench(poses, timestamps_s, half_extents, gravity_world, mass_kg=1.0,
                    inertia_per_mass=None, com_object=None, smoothing_window_s=0.20):
    """Return force/torque required FROM CONTACTS at the centre of mass.

    Default inertia is a uniform solid cuboid, an explicit approximation for
    an unknown box. Supply measured inertia/COM for a different distribution.
    """
    poses, times = validate_poses(poses), validate_times(timestamps_s)
    half, gravity = np.asarray(half_extents, float), np.asarray(gravity_world, float)
    if len(poses) != len(times) or half.shape != (3,) or np.any(half <= 0) or not np.isfinite(half).all():
        raise ValueError("invalid frame count or cuboid dimensions")
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("explicit gravity vector and positive mass are required")
    com = np.zeros(3) if com_object is None else np.asarray(com_object, float)
    if com.shape != (3,) or not np.isfinite(com).all():
        raise ValueError("COM must be a finite object-local 3-vector")
    if inertia_per_mass is None:
        inertia_per_mass = np.diag([(half[1]**2 + half[2]**2)/3,
                                   (half[0]**2 + half[2]**2)/3,
                                   (half[0]**2 + half[1]**2)/3])
    inertia = np.asarray(inertia_per_mass, float) * mass_kg
    if (inertia.shape != (3, 3) or not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia).min() <= 0):
        raise ValueError("inertia must be symmetric positive definite")
    r = poses[:, :3, :3]
    centers = poses[:, :3, 3] + np.einsum("tij,j->ti", r, com)
    acceleration = local_polynomial(times, centers, smoothing_window_s, derivative=2)
    # R_next R_current^T is an incremental rotation in WORLD coordinates.
    angular_mid = Rotation.from_matrix(r[1:] @ r[:-1].transpose(0, 2, 1)).as_rotvec() / np.diff(times)[:, None]
    mid_times = (times[:-1] + times[1:]) / 2
    omega_samples = np.stack([np.interp(times, mid_times, angular_mid[:, j]) for j in range(3)], axis=1)
    omega_samples[0] = angular_mid[0] + (times[0]-mid_times[0]) * (angular_mid[1]-angular_mid[0]) / (mid_times[1]-mid_times[0])
    omega_samples[-1] = angular_mid[-1] + (times[-1]-mid_times[-1]) * (angular_mid[-1]-angular_mid[-2]) / (mid_times[-1]-mid_times[-2])
    omega = local_polynomial(times, omega_samples, smoothing_window_s, derivative=0)
    alpha = local_polynomial(times, omega_samples, smoothing_window_s, derivative=1)
    inertia_world = r @ inertia @ r.transpose(0, 2, 1)
    momentum = np.einsum("tij,tj->ti", inertia_world, omega)
    torque = np.einsum("tij,tj->ti", inertia_world, alpha) + np.cross(omega, momentum)
    force = mass_kg * (acceleration - gravity)
    return dict(wrench_world=np.c_[force, torque], com_world=centers,
                acceleration_world=acceleration, omega_world=omega, alpha_world=alpha)


def friction_rays(normal, friction, count=8):
    normal = np.asarray(normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
        raise ValueError("contact normal must be finite and nonzero")
    if not np.isfinite(friction) or friction < 0 or count < 4:
        raise ValueError("invalid friction coefficient or cone discretization")
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = np.cross(normal, axis)
    tangent /= np.linalg.norm(tangent)
    other = np.cross(normal, tangent)
    theta = np.arange(count) * (2*np.pi/count)
    return normal + friction * (np.cos(theta)[:, None] * tangent + np.sin(theta)[:, None] * other)


def solve_contact_forces(candidates, wrench_world, com_world, characteristic_length_m,
                         regularization=1e-5, ambiguity_check=True, tolerance=0.08):
    """Nonnegative friction-cone fit, with contact-removal evidence.

    A force-bearing candidate is not necessarily identifiable: `necessary`
    additionally requires that removing its entire REGION makes a previously
    feasible wrench fit infeasible. Residuals are normalized force-equivalents.
    """
    wrench = np.asarray(wrench_world, float)
    center = np.asarray(com_world, float)
    if (wrench.shape != (6,) or center.shape != (3,) or not np.isfinite(wrench).all()
            or not np.isfinite(center).all() or characteristic_length_m <= 0
            or not np.isfinite(characteristic_length_m) or regularization < 0 or tolerance <= 0):
        raise ValueError("invalid wrench fit inputs")
    scale = np.r_[np.ones(3), np.full(3, 1/characteristic_length_m)]
    target = wrench * scale
    normalizer = max(np.linalg.norm(target), 1.0)
    columns, rays_by_candidate, penalty = [], [], []
    for candidate in candidates:
        point = np.asarray(candidate.point_world, float)
        if point.shape != (3,) or not np.isfinite(point).all() or not 0 <= candidate.prior <= 1:
            raise ValueError("invalid force candidate point or visual prior")
        rays = friction_rays(candidate.normal_world, candidate.friction)
        rays_by_candidate.append(rays)
        moments = np.cross(point - center, rays)
        columns.append(np.c_[rays, moments].T * scale[:, None])
        penalty.extend([1 + 4*(1-candidate.prior)] * len(rays))
    matrix = np.concatenate(columns, axis=1) if columns else np.zeros((6, 0))

    def fit(keep, regularized):
        a = matrix[:, keep]
        if not a.shape[1]:
            return np.empty(0), np.linalg.norm(target)/normalizer
        if regularized:
            a_fit = np.r_[a, np.diag(np.sqrt(regularization*np.asarray(penalty)[keep]))]
            b_fit = np.r_[target, np.zeros(a.shape[1])]
        else:
            a_fit, b_fit = a, target
        result = lsq_linear(a_fit, b_fit, bounds=(0, np.inf), tol=1e-10, max_iter=200)
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"contact force solve failed: {result.message}")
        return result.x, np.linalg.norm(a @ result.x-target)/normalizer

    coefficients, residual = fit(np.ones(matrix.shape[1], bool), True)
    forces = np.asarray([coefficients[8*i:8*i+8] @ rays for i, rays in enumerate(rays_by_candidate)]).reshape(-1, 3)
    feasible = residual <= tolerance
    removal = {}
    if ambiguity_check:
        for region in sorted({c.region for c in candidates}):
            keep = np.repeat([c.region != region for c in candidates], 8)
            removal[region] = float(fit(keep, False)[1])
    necessary = np.asarray([feasible and removal.get(c.region, 0) > tolerance for c in candidates], bool)
    return dict(forces_world=forces, normalized_residual=float(residual), feasible=bool(feasible),
                necessary=necessary, residual_without_region=removal,
                explained_wrench_world=(matrix @ coefficients)/scale)


def floor_candidates(pose, half_extents, normal_world, plane_offset_m, max_gap_m=0.03, friction=0.6):
    """Scene support candidates prevent attributing resting-box weight to a human."""
    normal = np.asarray(normal_world, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1):
        raise ValueError("floor normal must be a world-space unit vector")
    corners = np.array(list(product([-1, 1], repeat=3))) * np.asarray(half_extents)
    world = corners @ pose[:3, :3].T + pose[:3, 3]
    gap = world @ normal - plane_offset_m
    return [ForceCandidate("scene_floor", point, normal, friction=friction, source="scene")
            for point, distance in zip(world, gap) if abs(distance) <= max_gap_m]
