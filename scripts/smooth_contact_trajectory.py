#!/usr/bin/env python3
"""Smooth an IK trajectory, retain contact targets and remeasure all geometry.

This removes temporal jitter; it is not a dynamic feasibility certificate.
Quaternions are sign-aligned before normalized filtering. Foot/floor projections
are explicit and their effects are included in the final derivative statistics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_ik import ContactIK,point_jacobian,distance_jacobian
from prism_reconstruction.robot_physics import reference_velocities


def derivative_summary(model,q,times):
    velocity=reference_velocities(model,q,times);acceleration=np.gradient(velocity,times,axis=0)
    fields=dict(root_linear_speed_mps=np.linalg.norm(velocity[:,:3],axis=1),
        root_angular_speed_radps=np.linalg.norm(velocity[:,3:6],axis=1),
        root_acceleration_mps2=np.linalg.norm(acceleration[:,:3],axis=1),
        maximum_joint_speed_radps=abs(velocity[:,6:35]).max(axis=1),
        maximum_joint_acceleration_radps2=abs(acceleration[:,6:35]).max(axis=1),
        object_acceleration_mps2=np.linalg.norm(acceleration[:,35:38],axis=1))
    return {name:dict(zip(('median','p95','maximum'),np.quantile(values,[.5,.95,1]).tolist())) for name,values in fields.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sigma-seconds',type=float,default=.10)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not np.isfinite(args.sigma_seconds) or args.sigma_seconds<=0:raise ValueError('Positive smoothing width required')
    with np.load(args.input/'trajectory.npz') as a:arrays=dict(a)
    records=json.loads((args.input/'diagnostics.json').read_text());meta=json.loads((args.input/'metadata.json').read_text())
    solver=ContactIK(args.input/'scene.xml');m,d=solver.model,solver.data
    q=arrays['qpos'].copy();times=arrays['timestamps_s'];dt=np.median(np.diff(times))
    before=derivative_summary(m,q,times)
    for start in (3,solver.box_qadr+3):
        for frame in range(1,len(q)):
            if q[frame-1,start:start+4]@q[frame,start:start+4]<0:q[frame,start:start+4]*=-1
    q=gaussian_filter1d(q,args.sigma_seconds/dt,axis=0,mode='nearest')
    for start in (3,solver.box_qadr+3):q[:,start:start+4]/=np.linalg.norm(q[:,start:start+4],axis=1)[:,None]
    root_shift=[];box_shift=[];penetrations=[];errors=[]
    for frame in range(len(q)):
        d.qpos[:]=q[frame];mujoco.mj_forward(m,d)
        sole=[]
        for side,active in zip(('left','right'),arrays['stance'][frame]):
            if not active:continue
            body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
            for local in ([.12,.03,-.035],[.12,-.03,-.035],[-.05,0,-.035]):
                point,_=point_jacobian(m,d,body,np.asarray(local));sole.append(point[2])
        shift=-min(sole) if sole else 0.
        q[frame,2]+=shift;root_shift.append(shift)
        R=d.xmat[solver.box_body].reshape(3,3);half=m.geom_size[solver.box_geom]
        minimum=float(np.abs(R[2])@half);old=q[frame,solver.box_qadr+2]
        nominal=arrays['nominal_object_poses'][frame]
        nominal_height=nominal[2,3]-np.abs(nominal[2,:3])@half
        q[frame,solver.box_qadr+2]=minimum if nominal_height<.01 else max(minimum,old)
        box_shift.append(q[frame,solver.box_qadr+2]-old)
        d.qpos[:]=q[frame];mujoco.mj_forward(m,d)
        record=records[frame];record['smoothing_sigma_seconds']=args.sigma_seconds
        record.pop('stance_feet',None);record.pop('skeleton_error_mean_m',None)
        for target in record['targets']:
            robot,_=point_jacobian(m,d,target['robot_body_id'],np.asarray(target['robot_point_local']))
            box,_=point_jacobian(m,d,solver.box_body,np.asarray(target['point_object']))
            target['position_error_m']=float(np.linalg.norm(robot-box));errors.append(target['position_error_m'])
            original=np.asarray(target.get('point_object_original',target['point_object']))
            local=d.xmat[solver.box_body].reshape(3,3).T@(robot-d.xpos[solver.box_body])
            target['original_position_error_m']=float(np.linalg.norm(local-original))
            target['normal_error_m']=float(abs((local-original)@target['outward_normal_object']))
        collisions=[]
        for a,b in solver.pairs:
            distance,_=distance_jacobian(m,d,a,b,.02)
            if distance<-.002:collisions.append(dict(first=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,a),
                second=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,b),penetration_m=-distance))
        record['collision_violations']=collisions
        record['maximum_sampled_surface_penetration_m']=max([max(0.,-v) for v,_,_ in solver.box_surface_constraints()],default=0.)
        record['solver_status']='postprocessed_and_remeasured';record['termination']='temporal_filter';record['iterations']=0
        penetrations.append(max([x['penetration_m'] for x in collisions],default=0.))
    arrays['qpos']=q;arrays['joint_positions']=q[:,solver.qadr]
    arrays['object_position']=q[:,solver.box_qadr:solver.box_qadr+3];arrays['object_quaternion_wxyz']=q[:,solver.box_qadr+3:solver.box_qadr+7]
    arrays['smoothing_root_floor_shift_m']=np.asarray(root_shift);arrays['smoothing_object_floor_shift_m']=np.asarray(box_shift)
    meta.update(source_unsmoothed_trajectory_sha256=hashlib.sha256((args.input/'trajectory.npz').read_bytes()).hexdigest(),
        temporal_postprocessing=dict(sigma_seconds=args.sigma_seconds,root_floor_projection=True,
            object_floor_projection='Floor support follows nominal reconstructed bottom height <1 cm.',
            contact_points='Kept from the source retarget, not reoptimized or silently moved after filtering.'),
        derivatives_before=before,derivatives_after=derivative_summary(m,q,times),
        contact_error_median_m=float(np.median(errors)),contact_error_p95_m=float(np.quantile(errors,.95)),
        maximum_penetration_m=float(max(penetrations)),frames_penetrating_1cm=int(sum(x>.01 for x in penetrations)),
        postprocessing_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rl_ready=False)
    # These old optimization values do not describe the filtered trajectory.
    meta.pop('skeleton_error_mean_m',None)
    args.output.mkdir(parents=True);np.savez_compressed(args.output/'trajectory.npz',**arrays)
    shutil.copyfile(args.input/'scene.xml',args.output/'scene.xml')
    (args.output/'diagnostics.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    print(json.dumps({key:meta[key] for key in ('contact_error_median_m','contact_error_p95_m','maximum_penetration_m','frames_penetrating_1cm','derivatives_after')},indent=2))


if __name__=='__main__':main()
