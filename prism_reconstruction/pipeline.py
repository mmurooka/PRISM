"""Run visual/geometric contact inference and independent inverse-dynamics checks."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from .contacts import ContactConfig, infer_contacts
from .dynamics import ForceCandidate, floor_candidates, required_wrench, solve_contact_forces


def human_force_candidates(contact, scores, regions, pose, frame, config, friction):
    gap = contact['signed_gap_m'][frame]
    possible = (gap >= -config.max_penetration_m) & (gap <= config.leave_distance_m)
    normals = contact['outward_normals_object'][frame]
    points = contact['object_points'][frame]
    result, patches = [], []
    for region in np.unique(regions):
        for normal in np.unique(normals[possible & (regions == region)], axis=0):
            ids = np.flatnonzero(possible & (regions == region) & (normals == normal).all(axis=1))
            if len(ids) < config.min_region_vertices:
                continue
            prior = float(scores[frame, ids].mean())
            point = np.average(points[ids], axis=0, weights=scores[frame, ids] + .05)
            result.append(ForceCandidate(str(region), pose[:3, :3] @ point + pose[:3, 3],
                                         -pose[:3, :3] @ normal, prior, friction))
            patches.append(ids)
    return result, patches


def reconstruct_contacts(data, config=ContactConfig(), friction=.6, mass_kg=None,
                         temporal_force_smoothness=0.0, smoothing_window_s=.20):
    for field in ('world_frame', 'scale_source', 'source_video_sha256', 'object_id'):
        if field not in data or not str(np.asarray(data[field]).item()).strip():
            raise ValueError(f'missing provenance field: {field}')
    if str(np.asarray(data.get('length_unit', '')).item()) != 'm':
        raise ValueError('length_unit must explicitly be m; establish metric scale before contact inference')
    times, poses = data['timestamps_s'], data['world_from_object']
    vertices, scores = data['human_vertices_world'], data['contact_scores']
    regions = np.asarray(data['human_vertex_regions'], dtype=str)
    half = np.asarray(data['box_half_extents'], float)
    contact = infer_contacts(vertices, poses, half, scores, regions, times, config)
    if np.any(np.diff(times) > config.max_sample_gap_s):
        raise ValueError('motion has timestamp gaps too large for inverse dynamics; process continuous segments')
    normalized_mass = 1.0 if mass_kg is None else mass_kg
    motion = required_wrench(poses, times, half, data['gravity_world'], normalized_mass,
                             inertia_per_mass=data.get('inertia_per_mass'), com_object=data.get('com_object'),
                             smoothing_window_s=smoothing_window_s)
    if ('floor_normal_world' in data) != ('floor_offset_m' in data):
        raise ValueError('floor requires both world normal and plane offset')
    names = contact['region_names']
    necessary = np.zeros((len(times), len(names)), bool)
    possible = np.zeros_like(necessary)
    residual = np.empty(len(times))
    hand_only_residual = np.empty(len(times))
    records = []
    candidate_sequence = []
    length = float(np.linalg.norm(2*half))
    for t, pose in enumerate(poses):
        candidates, patches = human_force_candidates(contact, scores, regions, pose, t, config, friction)
        if 'floor_normal_world' in data:
            floor = floor_candidates(pose, half, data['floor_normal_world'], float(data['floor_offset_m']), friction=friction)
            candidates += floor
            patches += [np.empty(0, dtype=int) for _ in floor]
        fit = solve_contact_forces(candidates, motion['wrench_world'][t], motion['com_world'][t], length)
        candidate_sequence.append(candidates)
        hand_candidates = [c for c in candidates if c.source == 'scene' or c.region in ('left_hand', 'right_hand')]
        hand_fit = solve_contact_forces(hand_candidates, motion['wrench_world'][t], motion['com_world'][t], length,
                                        ambiguity_check=False)
        residual[t], hand_only_residual[t] = fit['normalized_residual'], hand_fit['normalized_residual']
        details = []
        for k, (candidate, force, patch) in enumerate(zip(candidates, fit['forces_world'], patches)):
            magnitude = float(np.linalg.norm(force))
            selected = fit['feasible'] and magnitude > .05*normalized_mass*float(np.linalg.norm(data['gravity_world']))
            ids = np.flatnonzero(names == candidate.region)
            if candidate.source == 'human' and len(ids):
                possible[t, ids[0]] |= selected
                necessary[t, ids[0]] |= bool(fit['necessary'][k])
            details.append(dict(region=candidate.region, source=candidate.source,
                                vertex_ids=patch.tolist(), visual_prior=candidate.prior,
                                point_object=((candidate.point_world-pose[:3,3]) @ pose[:3,:3]).tolist(),
                                normal_world=candidate.normal_world.tolist(),
                                force_world=force.tolist(), force_object=(pose[:3,:3].T @ force).tolist(),
                                explanatory=bool(selected), necessary=bool(fit['necessary'][k])))
        records.append(dict(frame=t, time_s=float(times[t]), feasible=fit['feasible'],
                            normalized_residual=float(residual[t]), hand_only_residual=float(hand_only_residual[t]),
                            residual_without_region=fit['residual_without_region'], candidates=details))
    metadata = dict(schema='whole_body_hoi_contacts_v1', contact_config=asdict(config),
                    friction_coefficient=friction, mass_kg=mass_kg,
                    derivative_smoothing_window_s=smoothing_window_s,
                    temporal_force_smoothness=temporal_force_smoothness,
                    force_unit='N_per_kg' if mass_kg is None else 'N',
                    torque_unit='Nm_per_kg' if mass_kg is None else 'Nm',
                    inertia_assumption='provided' if 'inertia_per_mass' in data else 'uniform_solid_cuboid',
                    scene_support='floor_plane' if 'floor_normal_world' in data else 'not_provided',
                    shape_assumption=str(np.asarray(data.get('shape_assumption', 'provided_cuboid')).item()),
                    contact_mode_codes={'0':'no_visual_geometric_contact','1':'sticking','2':'sliding'},
                    interpretation='Explanatory forces and model-conditional necessity, not measured forces or uniquely identified contacts.',
                    intervals=contact.pop('intervals'))
    metadata.update({k:str(np.asarray(data[k]).item()) for k in ('world_frame','scale_source','source_video_sha256','object_id')})
    metadata['contact_prior_source']=str(np.asarray(data.get('contact_prior_source','provided_unspecified')).item())
    arrays = dict(**contact, **motion, timestamps_s=times, world_from_object=poses, box_half_extents=half,
                  human_vertex_regions=regions, dynamics_possible=possible, dynamics_necessary=necessary,
                  normalized_wrench_residual=residual, hand_only_wrench_residual=hand_only_residual)
    if not np.isfinite(temporal_force_smoothness) or temporal_force_smoothness<0:
        raise ValueError('Temporal force smoothness must be finite and nonnegative')
    if temporal_force_smoothness:
        from .temporal_forces import solve_force_sequence
        sequence=solve_force_sequence(candidate_sequence,motion['wrench_world'],motion['com_world'],times,
                                       length,smoothness=temporal_force_smoothness)
        arrays.update(temporal_region_names=np.asarray([name[1] for name in sequence['region_names']]),
                      temporal_region_sources=np.asarray([name[0] for name in sequence['region_names']]),
                      temporal_region_forces_world=sequence['region_forces_world'],
                      temporal_wrench_residual=sequence['normalized_residual'],
                      temporal_force_feasible=sequence['feasible'])
        for frame,forces in zip(records,sequence['forces_world']):
            frame['temporal_force_feasible']=bool(sequence['feasible'][frame['frame']])
            for candidate,force in zip(frame['candidates'],forces):
                candidate['temporal_force_world']=force.tolist()
        metadata['temporal_force_interpretation']='Smoothed explanatory forces; contact necessity remains the independent per-frame region-removal test.'
    return arrays, metadata, records


def run_archive(source, output, **kwargs):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to mix a new reconstruction with existing outputs: {output}')
    with np.load(source, allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    arrays, metadata, records = reconstruct_contacts(data, **kwargs)
    metadata['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata['input_file'] = str(source.resolve())
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'contacts.npz', **arrays)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (output/'force_evidence.json').write_text(json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return metadata
