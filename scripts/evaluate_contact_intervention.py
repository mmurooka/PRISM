#!/usr/bin/env python3
"""Compare identical policy states with and without one body-box collision pair."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_tracking_env import ContactTrackingEnv
from prism_reconstruction.contact_intervention import disable_box_weld_contact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--region',default='right_thigh')
    parser.add_argument('--start-time',type=float,default=4.79)
    parser.add_argument('--intervention-time',type=float,default=.08)
    parser.add_argument('--horizon',type=float,default=.6)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not 0<args.intervention_time<args.horizon:raise ValueError('Intervention must be inside the rollout')
    torch.set_num_threads(1)
    folder=args.checkpoint.parent
    if folder.name=='checkpoints':folder=folder.parent
    training=json.loads((folder/'metadata.json').read_text())
    policy=PPO.load(args.checkpoint,device='cpu');args.output.mkdir(parents=True)
    initial_states=[];branches={}
    for remove in (False,True):
        name='leg_contact_disabled' if remove else 'normal_contacts';out=args.output/name;out.mkdir()
        env=ContactTrackingEnv(training['reference'],folder/'physics.xml',horizon_s=args.horizon,random_start=False,
            action_mode=training['action_mode'],teacher_checkpoint=training.get('teacher_checkpoint'),
            teacher_residual_scale=training.get('teacher_residual_scale',1.),
            policy_control=training.get('policy_control','residual'),pd_gain_source=training.get('pd_gain_source','local'),
            contact_reward=training.get('contact_reward','additive'),contact_group=training.get('contact_group',[]),
            contact_group_weight=training.get('contact_group_weight',0.),
            inverse_torque_reference=training.get('inverse_torque_reference'),record_contact_details=True,
            contact_force_location=training.get('contact_force_location','body'))
        steps=args.intervention_time/env.control_dt
        if not np.isclose(steps,round(steps)):raise ValueError('Intervention must align with control steps')
        intervention_step=int(round(steps));observation,_=env.reset(seed=0,options={'start_time':args.start_time})
        _,_,frame=env._reference(args.start_time+args.intervention_time)
        welds={int(env.model.body_weldid[t['robot_body_id']]) for t in env.records[frame]['targets'] if t['region']==args.region}
        if len(welds)!=1:raise ValueError('Exactly one active articulated body is required for the region')
        weld=next(iter(welds));qpos=[];times=[];frames=[];diagnostics=[];metrics=[];intervention=None
        for step in range(int(np.ceil(args.horizon/env.control_dt))):
            if step==intervention_step:
                initial_states.append(np.r_[env.data.qpos,env.data.qvel,env.previous_action,observation])
                if remove:
                    intervention=disable_box_weld_contact(env.model,env.box_geom,weld)
                    mujoco.mj_forward(env.model,env.data)
            phase=env.start_time+env.elapsed;_,_,frame=env._reference(phase)
            qpos.append(env.data.qpos.copy());times.append(phase);frames.append(frame)
            diagnostics.append(dict(frame=frame,targets=env.records[frame]['targets']))
            observation,_,terminated,truncated,info=env.step(policy.predict(observation,deterministic=True)[0])
            if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):raise ValueError('Unexpected external assistance')
            info['selected_body_upward_force_n']=float(sum(c['force_on_object_world_n'][2] for c in info['object_contacts'] if c['other_weld']==weld))
            metrics.append(info)
            if terminated or truncated:break
        if len(initial_states)!=1+int(remove):raise ValueError('Branch terminated before the intervention')
        np.savez_compressed(out/'trajectory.npz',qpos=qpos,timestamps_s=times,frame_indices=frames)
        (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        mujoco.mj_saveLastXML(str(out/'scene.xml'),env.model)
        metadata=dict(schema='g1_unassisted_physics_rollout_v1',source_video_sha256=env.meta['source_video_sha256'],
            independent_diagnostic_frames=False,controller='contact_collision_ablation' if remove else 'residual_reference_PD',
            reference=training['reference'],checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            robot_root_assistance=False,object_assistance=False,safe_for_robot_execution=False,
            start_time_s=args.start_time,intervention_time_s=args.intervention_time,region=args.region,
            intervention=intervention,duration_s=env.elapsed,
            interpretation='Artificial simulation-only collision counterfactual; not a human contact annotation or a physical-robot experiment.')
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2));branches[name]=metrics
    error=float(np.max(abs(initial_states[0]-initial_states[1])))
    if error!=0:raise AssertionError('Pre-intervention states differ')
    first=branches['normal_contacts'];second=branches['leg_contact_disabled'];count=min(len(first),len(second))
    poses=[]
    for branch in ('normal_contacts','leg_contact_disabled'):
        with np.load(args.output/branch/'trajectory.npz') as archive:poses.append(archive['qpos'][count-1,env.box_q:env.box_q+7])
    rotation_difference=2*np.arccos(np.clip(abs(poses[0][3:]@poses[1][3:]),0,1))
    summary=dict(pre_intervention_state_max_error=error,common_duration_s=count*.02,
        common_pose_snapshot_elapsed_s=(count-1)*env.control_dt,
        box_position_difference_m=float(np.linalg.norm(poses[0][:3]-poses[1][:3])),
        box_rotation_difference_degrees=float(np.rad2deg(rotation_difference)),
        box_height_difference_normal_minus_disabled_m=first[count-1]['box_height_m']-second[count-1]['box_height_m'],
        selected_body_upward_impulse_normal_Ns=.02*sum(r['selected_body_upward_force_n'] for r in first[intervention_step:count]),
        selected_body_upward_impulse_disabled_Ns=.02*sum(r['selected_body_upward_force_n'] for r in second[intervention_step:count]),
        normal_duration_s=len(first)*.02,disabled_duration_s=len(second)*.02,
        code_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),Path(__file__).resolve().parents[1]/'prism_reconstruction/contact_intervention.py')},
        interpretation='Total closed-loop effect of disabling the selected box-body collision, conditioned on this simulator, policy and initial state. Other collision masks, masses and motor control remain unchanged.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
