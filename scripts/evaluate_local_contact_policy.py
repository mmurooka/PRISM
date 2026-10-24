#!/usr/bin/env python3
"""Evaluate zero-residual PD or a local PPO policy in unassisted physics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_tracking_env import ContactTrackingEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--physics-scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--teacher-checkpoint',type=Path)
    parser.add_argument('--teacher-residual-scale',type=float,default=1.)
    parser.add_argument('--start-times',type=float,nargs='+',default=[0.,2.,4.,4.79,8.])
    parser.add_argument('--horizon',type=float,default=12.)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    action_mode='joint_offset'
    if args.checkpoint:
        folder=args.checkpoint.parent
        if folder.name=='checkpoints':folder=folder.parent
        training_metadata=json.loads((folder/'metadata.json').read_text())
        action_mode=training_metadata.get('action_mode','joint_offset')
    teacher=args.teacher_checkpoint
    if args.checkpoint:
        teacher=teacher or training_metadata.get('teacher_checkpoint')
    residual_scale=training_metadata.get('teacher_residual_scale',1.) if args.checkpoint else args.teacher_residual_scale
    contact_reward=training_metadata.get('contact_reward','additive') if args.checkpoint else 'additive'
    policy_control=training_metadata.get('policy_control','residual') if args.checkpoint else 'residual'
    env=ContactTrackingEnv(args.reference,args.physics_scene,horizon_s=args.horizon,random_start=False,
        action_mode=action_mode,teacher_checkpoint=teacher,teacher_residual_scale=residual_scale,
        contact_reward=contact_reward,policy_control=policy_control)
    policy=None
    if args.checkpoint:
        from stable_baselines3 import PPO
        policy=PPO.load(args.checkpoint,device='cpu')
    args.output.mkdir(parents=True);summaries=[]
    for episode,start in enumerate(args.start_times):
        output=args.output/f'phase_{episode:02d}';output.mkdir()
        observation,_=env.reset(seed=args.seed+episode,options={'start_time':start})
        qpos=[];times=[];frames=[];diagnostics=[];metrics=[];rewards=[]
        for step in range(int(args.horizon/env.control_dt)+1):
            phase=env.start_time+env.elapsed;_,_,frame=env._reference(phase)
            qpos.append(env.data.qpos.copy());times.append(phase);frames.append(frame)
            diagnostics.append(dict(frame=frame,targets=env.records[frame]['targets']))
            action=policy.predict(observation,deterministic=True)[0] if policy else np.zeros(29)
            observation,reward,terminated,truncated,info=env.step(action)
            metrics.append(info);rewards.append(reward)
            if terminated or truncated:break
        np.savez_compressed(output/'trajectory.npz',qpos=np.asarray(qpos),timestamps_s=np.asarray(times),frame_indices=np.asarray(frames))
        (output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
        shutil.copyfile(args.physics_scene,output/'scene.xml')
        summary=dict(episode=episode,start_time_s=start,duration_s=env.elapsed,
            completed_remaining_clip=not terminated and env.start_time+env.elapsed>=env.times[-1],
            terminated=terminated,truncated=truncated,return_=float(sum(rewards)),
            object_error_mean_m=float(np.mean([m['object_error_m'] for m in metrics])),
            object_error_final_m=metrics[-1]['object_error_m'],
            failure_reasons=metrics[-1]['failure_reasons'],
            contact_force_fraction_mean=float(np.mean([m['contact_force_fraction'] for m in metrics if m['active_contact_targets']]))
                if any(m['active_contact_targets'] for m in metrics) else None)
        summary['regions']={}
        for region in sorted(set().union(*(m['region_object_contact_force_n'] for m in metrics))):
            selected=[m for m in metrics if region in m['region_object_contact_force_n']]
            summary['regions'][region]=dict(active_steps=len(selected),
                force_present_fraction=float(np.mean([m['region_object_contact_force_n'][region]>1. for m in selected])),
                force_and_gap_below_3cm_fraction=float(np.mean([m['region_object_contact_force_n'][region]>1.
                    and m['region_target_gap_m'][region]<.03 for m in selected])),
                target_gap_median_m=float(np.median([m['region_target_gap_m'][region] for m in selected])))
        metadata=dict(schema='g1_unassisted_physics_rollout_v1',source_video_sha256=env.meta['source_video_sha256'],
            independent_diagnostic_frames=False,controller=('finetuned_PRISM_teacher' if policy_control=='teacher_absolute' else ('teacher_plus_residual_PPO' if teacher else 'residual_PPO')) if policy else ('released_PRISM_teacher' if teacher else 'zero_residual_PD'),
            action_mode=action_mode,
            teacher_residual_scale=residual_scale,
            contact_reward=contact_reward,
            policy_control=policy_control,
            teacher_checkpoint_sha256=env.teacher.sha256 if env.teacher is not None else None,
            evaluation_code_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__),Path(__file__).resolve().parents[1]/'prism_reconstruction/contact_tracking_env.py')},
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
            robot_root_assistance=False,object_assistance=False,object_mass_kg=float(env.model.body_mass[env.box]),
            safe_for_robot_execution=False,**summary)
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))


if __name__=='__main__':main()
