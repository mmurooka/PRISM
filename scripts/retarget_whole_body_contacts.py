#!/usr/bin/env python3
"""Retarget local HOI reconstruction to 29-DoF G1 with soft surface contacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d,binary_closing,binary_opening

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_ik import ContactIK,make_scene
from prism_reconstruction.retargeting import transform_demonstration,select_contact_targets
from prism_reconstruction.geometry import smpl_vertex_regions


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('human','object','contacts','robot-xml','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--policy',choices=('visual_or_necessary','all_explanatory','hands_only'),default='all_explanatory')
    parser.add_argument('--frames',type=int,nargs='+',help='Independent diagnostic frames; no temporal continuity claim')
    parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--contact-weight',type=float,default=120.)
    parser.add_argument('--warm-start',type=Path,help='Existing matching trajectory for collision-refinement initialization')
    args=parser.parse_args()
    code_hashes={name:sha(Path(__file__).resolve().parents[1]/'prism_reconstruction'/name)
                 for name in ('contact_ik.py','retargeting.py')}
    if args.output.exists():raise FileExistsError(args.output)
    if args.iterations<1:raise ValueError('Positive iteration count required')
    with np.load(args.human) as a:human=dict(a)
    with np.load(args.object) as a:box=dict(a)
    with np.load(args.contacts/'contacts.npz') as a:contacts=dict(a)
    meta=json.loads((args.contacts/'metadata.json').read_text())
    records=json.loads((args.contacts/'force_evidence.json').read_text())
    for key in ('world_frame','source_video_sha256'):
        if str(human[key])!=str(box[key]) or str(human[key])!=meta[key]:raise ValueError(f'Mismatched {key}')
    for field in ('timestamps_s','world_from_object','box_half_extents'):
        if not np.allclose(box[field],contacts[field]):raise ValueError(f'Mismatched contacts {field}')
    if not np.allclose(human['timestamps_s'],box['timestamps_s']):raise ValueError('Mismatched human timestamps')
    for t in range(len(records)):
        reconstructed=contacts['human_points_object'][t]@box['world_from_object'][t,:3,:3].T+box['world_from_object'][t,:3,3]
        if not np.allclose(reconstructed,human['human_vertices_world'][t],atol=1e-5):
            raise ValueError('Contact archive belongs to a different human reconstruction')
    if not np.all(box['valid']) or str(human['length_unit'])!='m':raise ValueError('Valid metric object geometry required')
    args.output.mkdir(parents=True)
    scene=make_scene(args.robot_xml,box['box_half_extents'],args.output/'scene.xml')
    solver=ContactIK(scene,args.contact_weight)
    warm_start=None
    if args.warm_start:
        warm_meta=json.loads((args.warm_start/'metadata.json').read_text())
        if warm_meta['source_human_sha256']!=sha(args.human) or warm_meta['source_object_sha256']!=sha(args.object):
            raise ValueError('Warm-start reconstruction mismatch')
        with np.load(args.warm_start/'trajectory.npz') as a:
            if not np.array_equal(a['frame_indices'],np.arange(len(records))):raise ValueError('Warm start must cover all frames')
            warm_start=a['qpos'].copy()
        if warm_start.shape!=(len(records),solver.model.nq):raise ValueError('Warm-start robot configuration mismatch')
    demonstration=transform_demonstration(human['human_joints_world'],box['world_from_object'],
        box['box_half_extents'],box['floor_normal_world'],float(box['floor_offset_m']),solver.body_scale(human['human_joints_world']))
    targets=select_contact_targets(records,contacts,meta,args.policy)
    # Preserve proximal/distal anatomy when a thigh patch moves to G1; simply
    # choosing the nearest robot vertex can map the human knee area to its hip.
    for frame,items in enumerate(targets):
        for item in items:
            if not item['region'].endswith('thigh'):continue
            candidate=min((c for c in records[frame]['candidates'] if c['region']==item['region']),
                key=lambda c:np.linalg.norm(np.asarray(c['point_object'])-item['point_object']))
            hip,knee=(1,4) if item['region'].startswith('left') else (2,5)
            origin=human['rest_joints'][hip];bone=human['rest_joints'][knee]-origin
            points=human['template_vertices'][candidate['vertex_ids']]
            item['limb_fraction']=float(np.clip(np.median((points-origin)@bone/(bone@bone)),0,1))
    labels=smpl_vertex_regions(human['template_vertices'],human['skinning_weights'],human['rest_joints'])
    stance=[];foot_heights=[]
    for side in ('left','right'):
        vertices=human['human_vertices_world'][:,labels==side+'_foot']
        height=np.quantile(vertices@box['floor_normal_world']-box['floor_offset_m'],.1,axis=1)
        centres=gaussian_filter1d(vertices.mean(axis=1),2,axis=0)
        speed=np.linalg.norm(np.gradient(centres,human['timestamps_s'],axis=0),axis=1)
        contact=(height<.06)&(speed<.45)
        # Fill only short gaps. Keep endpoint observations instead of the
        # morphology functions' default zero padding erasing the first stance.
        contact=binary_opening(binary_closing(np.pad(contact,4,mode='edge'),structure=np.ones(5)),structure=np.ones(3))[4:-4]
        stance.append(contact);foot_heights.append(height)
    stance=np.stack(stance,axis=1);foot_heights=np.stack(foot_heights,axis=1)
    fallback=(~stance.any(axis=1))&(foot_heights.min(axis=1)<.07)
    stance[np.flatnonzero(fallback),foot_heights[fallback].argmin(axis=1)]=True
    frames=np.arange(len(records)) if args.frames is None else np.asarray(args.frames)
    if np.any(frames<0) or np.any(frames>=len(records)) or len(np.unique(frames))!=len(frames):raise ValueError('Invalid frame selection')
    previous=None;trajectories=[];diagnostics=[];started=time.time()
    for frame in frames:
        independent=args.frames is not None
        if warm_start is not None:
            initial=warm_start[frame].copy()
            if independent:solver.anchors.clear();solver.foot_anchors.clear()
        elif previous is None or independent:
            initial=solver.neutral();initial[:3]=demonstration['human_joints_robot_world'][frame,0]
            solver.set_object(initial,demonstration['nominal_object_poses'][frame]);solver.anchors.clear();solver.foot_anchors.clear()
        else:initial=previous.copy()
        iterations=max(40,args.iterations) if previous is None or independent else args.iterations
        q,record=solver.solve(demonstration['human_joints_robot_world'][frame],demonstration['nominal_object_poses'][frame],
            targets[frame],initial,None if independent else previous,int(frame),iterations,
            float(np.median(np.diff(human['timestamps_s']))),stance[frame])
        trajectories.append(q);diagnostics.append(record);previous=q
        if len(trajectories)%10==0 or independent:
            print(json.dumps(dict(frame=int(frame),elapsed_s=round(time.time()-started,2),
                contacts=len(record['targets']),max_contact_error_m=max([t['position_error_m'] for t in record['targets']],default=0),
                max_penetration_m=max([x['penetration_m'] for x in record['collision_violations']],default=0))),flush=True)
    qpos=np.asarray(trajectories)
    m=solver.model
    joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in solver.hinges]
    np.savez_compressed(args.output/'trajectory.npz',qpos=qpos,joint_positions=qpos[:,solver.qadr],
        joint_names=np.asarray(joint_names),timestamps_s=human['timestamps_s'][frames],frame_indices=frames,
        object_position=qpos[:,solver.box_qadr:solver.box_qadr+3],object_quaternion_wxyz=qpos[:,solver.box_qadr+3:solver.box_qadr+7],
        stance=stance[frames],stance_lowest_foot_fallback=fallback[frames],**demonstration)
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False))
    errors=[t['position_error_m'] for r in diagnostics for t in r['targets']]
    penetrations=[max([x['penetration_m'] for x in r['collision_violations']],default=0) for r in diagnostics]
    metadata=dict(schema='g1_whole_body_contact_ik_v1',policy=args.policy,body_scale=demonstration['body_scale'],
        contact_weight=args.contact_weight,iterations_per_frame=args.iterations,
        warm_start_trajectory_sha256=sha(args.warm_start/'trajectory.npz') if args.warm_start else None,
        source_video_sha256=str(human['source_video_sha256']),source_human_sha256=sha(args.human),source_object_sha256=sha(args.object),
        source_contacts_sha256=sha(args.contacts/'contacts.npz'),source_robot_xml_sha256=sha(args.robot_xml),
        solver_code_sha256=code_hashes['contact_ik.py'],code_sha256=code_hashes,
        frames=len(frames),independent_diagnostic_frames=args.frames is not None,
        contact_error_median_m=float(np.median(errors)) if errors else None,
        contact_error_p95_m=float(np.quantile(errors,.95)) if errors else None,
        maximum_penetration_m=float(max(penetrations)),frames_penetrating_1cm=int(np.sum(np.array(penetrations)>.01)),
        skeleton_error_mean_m=float(np.mean([r['skeleton_error_mean_m'] for r in diagnostics])),
        elapsed_s=time.time()-started,object_mass_kg=1.,object_mass_measured=False,
        stance_method='SMPL foot lower-decile height <6 cm and smoothed speed <0.45 m/s; short-gap morphology; lowest-foot fallback below 7 cm for this walking sequence.',
        stance_fallback_frames=int(fallback[frames].sum()),
        method='Independent sequential tangent-space QP IK; joint robot/object pose optimization; object-local soft contact patches.',
        limitations=['Kinematics only: no balance, torque or dynamic feasibility guarantee.',
            'Self-collision checks exclude links within five edges in the source kinematic tree.',
            'Soft collision constraints can leave violations, which are reported explicitly.',
            'Explanatory contact assignments are hypotheses, not verified labels.',
            'Body motion is scaled by leg length; the reconstructed physical box dimensions are retained.',
            'Robot patch is fixed during consecutive evidence; object-side points may move with estimated contact.'],
        safe_for_robot_execution=False,rl_ready=False)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
