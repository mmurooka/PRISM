#!/usr/bin/env python3
"""Measure contact at every physics step and verify recording leaves motion unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_tracking_env import ContactTrackingEnv


def longest_run(flags, timestep):
    changes = np.diff(np.r_[False, flags, False].astype(int))
    lengths = np.flatnonzero(changes == -1)-np.flatnonzero(changes == 1)
    return float(max(lengths, default=0)*timestep)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-time', type=float, default=4.79)
    parser.add_argument('--horizon', type=float, default=.6)
    parser.add_argument('--regions', nargs='+', default=['right_thigh', 'right_knee'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.horizon) or args.horizon <= 0:
        raise ValueError('Positive finite horizon required')
    folder = args.checkpoint.parent
    if folder.name == 'checkpoints':
        folder = folder.parent
    meta = json.loads((folder/'metadata.json').read_text())
    env = ContactTrackingEnv(meta['reference'], folder/'physics.xml', horizon_s=args.horizon,
        random_start=False, action_mode=meta['action_mode'], teacher_checkpoint=meta.get('teacher_checkpoint'),
        policy_control=meta.get('policy_control', 'residual'), pd_gain_source=meta.get('pd_gain_source', 'local'),
        teacher_residual_scale=meta.get('teacher_residual_scale', 1.),
        inverse_torque_reference=meta.get('inverse_torque_reference'),
        contact_reward=meta.get('contact_reward', 'additive'),
        contact_group=meta.get('contact_group', []), contact_group_weight=meta.get('contact_group_weight', 0.),
        contact_force_location=meta.get('contact_force_location', 'body'), record_contact_details=True)
    policy = PPO.load(args.checkpoint, device='cpu')
    def rollout():
        obs, _ = env.reset(seed=0, options={'start_time': args.start_time})
        states = [env.data.qpos.copy()]
        velocities = [env.data.qvel.copy()]
        actions, diagnostics = [], []
        for _ in range(int(np.ceil(args.horizon/env.control_dt))+1):
            action = policy.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            states.append(env.data.qpos.copy())
            velocities.append(env.data.qvel.copy())
            actions.append(action.copy())
            diagnostics.append(info)
            if terminated or truncated:
                break
        return np.asarray(states), np.asarray(velocities), np.asarray(actions), diagnostics

    baseline = rollout()
    samples = []
    original = mujoco.mj_step
    def record_step(model, data):
        original(model, data)
        # mj_step advances qpos/time, while its contact forces and derived body
        # transforms still describe the pre-integration state. Do not call
        # mj_forward here: even a logging-only forward could affect warm starts.
        time = env.start_time+data.time-model.opt.timestep
        _, _, frame = env._reference(time)
        contacts = []
        for i in range(data.ncon):
            contact = data.contact[i]
            first, second = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if env.box not in (first, second):
                continue
            other = second if first == env.box else first
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            point = (contact.pos-data.xpos[env.box])@data.xmat[env.box].reshape(3, 3)
            contacts.append((int(model.body_weldid[other]), max(0., float(force[0])), point))
        row = []
        for region in args.regions:
            targets = [t for t in env.records[frame]['targets'] if t['region'] == region]
            gap, body_force, patch_force = float('inf'), 0., 0.
            for target in targets:
                robot, box = env._contact_positions(target)
                gap = min(gap, float(np.linalg.norm(robot-box)))
                weld = int(model.body_weldid[target['robot_body_id']])
                same = [(force, point) for body, force, point in contacts if body == weld]
                body_force = max(body_force, sum(force for force, _ in same))
                patch_force = max(patch_force, sum(force for force, point in same
                    if np.linalg.norm(point-target['point_object']) < .03))
            row.append((bool(targets), gap, body_force, patch_force))
        # Keep inactive samples too: separate contact episodes must never join.
        samples.append((time, row))
    mujoco.mj_step = record_step
    try:
        recorded = rollout()
    finally:
        mujoco.mj_step = original
        env.close()
    parity = {}
    for name, a, b in zip(('qpos', 'qvel', 'actions'), baseline[:3], recorded[:3]):
        if a.shape != b.shape:
            raise AssertionError('Recording changed rollout length')
        parity[name] = float(np.max(np.abs(a-b)))
        if not np.array_equal(a, b):
            raise AssertionError(f'Recording changed {name}: {parity[name]}')
    times = np.asarray([time for time, _ in samples])
    values = np.asarray([row for _, row in samples])
    dt = env.model.opt.timestep
    if len(times)>1 and not np.allclose(np.diff(times), dt, atol=1e-9):
        raise AssertionError('Physics contact sampling has gaps')
    summary = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
        physics_xml_sha256=sha(folder/'physics.xml'), reference=meta['reference'],
        reference_trajectory_sha256=sha(Path(meta['reference'])/'trajectory.npz'),
        start_time_s=args.start_time, requested_horizon_s=args.horizon,
        actual_duration_s=len(samples)*dt, physics_sample_rate_hz=1/dt,
        control_sample_rate_hz=1/env.control_dt, sample_count=len(samples),
        exact_recording_parity_max_abs_difference=parity, code_sha256=sha(__file__), regions={},
        interpretation='Simulated contact sampled at every physics step, not continuous-time or human ground truth. Physics samples use pre-integration forces/geometry; control samples are the existing post-control metric. No extra forward call or applied force is used.')
    for j, region in enumerate(args.regions):
        active = values[:, j, 0].astype(bool)
        gap, force, patch = values[:, j, 1:].T
        coarse = baseline[3]
        coarse_patch = np.array([x['region_patch_contact_force_n'].get(region, 0)>1 for x in coarse])
        summary['regions'][region] = dict(active_physics_samples=int(active.sum()),
            longest_body_force_contact_s=longest_run(force>1, dt),
            longest_proxy_and_force_contact_s=longest_run((gap<.03)&(force>1), dt),
            longest_actual_patch_contact_s=longest_run(patch>1, dt),
            actual_patch_positive_samples=int((patch>1).sum()),
            control_rate_longest_actual_patch_contact_s=longest_run(coarse_patch, env.control_dt))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'samples.npz', source_times_s=times, region_names=args.regions,
        target_active=values[:, :, 0].astype(bool), proxy_gap_m=values[:, :, 1],
        body_normal_force_n=values[:, :, 2], actual_patch_normal_force_n=values[:, :, 3])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    main()
