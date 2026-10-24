#!/usr/bin/env python3
"""Compare inverse-force feasibility within different teacher residual ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_tracking_env import ContactTrackingEnv
from prism_reconstruction.robot_physics import reference_velocities
from prism_reconstruction.inverse_contact_audit import audit_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','physics-scene','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scales',nargs='+',type=float,default=[1.,10.])
    parser.add_argument('--gap',type=float,default=.02)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    env=ContactTrackingEnv(args.reference,args.physics_scene,teacher_checkpoint=args.checkpoint)
    m,d=env.model,env.data;teacher=env.teacher
    velocity=reference_velocities(m,env.qref,env.times);acceleration=np.gradient(velocity,env.times,axis=0,edge_order=2)
    records=[]
    for i,time in enumerate(env.times[:-1]):
        env.reset(options={'start_time':float(time)})
        m.geom_margin[:]=args.gap;m.geom_gap[:]=0.
        d.qvel[:]=velocity[i]
        with torch.no_grad():action=teacher.actor(torch.tensor(teacher.observation(env)[None]))[0].numpy()
        result=dict(frame=i,time_s=float(time),scales={})
        for scale in args.scales:
            torque=[]
            for sign in (-1,1):
                desired=teacher.default+teacher.action_scale*np.clip(action+sign*scale,-teacher.clip,teacher.clip)
                torque.append(np.clip(env.motor['kp']*(desired-d.qpos[env.qadr])-env.motor['kd']*d.qvel[env.vadr],
                                     env.motor['lower'],env.motor['upper']))
            fit=audit_frame(m,d,env.qref[i],velocity[i],acceleration[i],args.gap,motor_bounds=torque)
            result['scales'][str(scale)]={key:fit[key] for key in ('robot_root_residual','box_residual','motor_residual','solver_success')}
        records.append(result)
    summaries={str(scale):dict(frames=len(records),necessary_fit_below_5percent_frames=sum(
        r['scales'][str(scale)]['solver_success'] and max(r['scales'][str(scale)][k] for k in
            ('robot_root_residual','box_residual','motor_residual'))<.05 for r in records)) for scale in args.scales}
    args.output.mkdir(parents=True);(args.output/'frames.json').write_text(json.dumps(records,indent=2))
    metadata=dict(summaries=summaries,maximum_admitted_gap_m=args.gap,
        teacher_checkpoint_sha256=teacher.sha256,
        source_trajectory_sha256=hashlib.sha256((args.reference/'trajectory.npz').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Optimistic necessary check at reference states, previous teacher actions reset to zero each frame. Not a rollout or success rate.',
        torque_intervals='Instantaneous PD torque reachable by bounded residual actions, intersected with unchanged physical effort limits.')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
