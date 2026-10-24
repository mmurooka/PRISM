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
    parser.add_argument('--start-times',type=float,nargs='+',default=[0.,2.,4.,4.79,8.])
    parser.add_argument('--horizon',type=float,default=12.)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    env=ContactTrackingEnv(args.reference,args.physics_scene,horizon_s=args.horizon,random_start=False)
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
            contact_force_fraction_mean=float(np.mean([m['contact_force_fraction'] for m in metrics if m['active_contact_targets']]))
                if any(m['active_contact_targets'] for m in metrics) else None)
        metadata=dict(schema='g1_unassisted_physics_rollout_v1',source_video_sha256=env.meta['source_video_sha256'],
            independent_diagnostic_frames=False,controller='residual_PPO' if policy else 'zero_residual_PD',
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
            robot_root_assistance=False,object_assistance=False,object_mass_kg=float(env.model.body_mass[env.box]),
            safe_for_robot_execution=False,**summary)
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))


if __name__=='__main__':main()
