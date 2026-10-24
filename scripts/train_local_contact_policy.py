#!/usr/bin/env python3
"""Exploratory PPO on one reconstructed demonstration, entirely local."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.robot_physics import prepare_physics_scene


def make_environment(reference,scene,mode,seed):
    from prism_reconstruction.contact_tracking_env import ContactTrackingEnv
    from stable_baselines3.common.monitor import Monitor
    env=Monitor(ContactTrackingEnv(reference,scene,reward_mode=mode,horizon_s=2.,random_start=True))
    env.reset(seed=seed)
    return env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=500000)
    parser.add_argument('--envs',type=int,default=16)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--reward-mode',choices=('whole_body','hands_only'),default='whole_body')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<1 or args.envs<1:raise ValueError('Positive training budget and environment count required')
    args.output.mkdir(parents=True)
    scene=prepare_physics_scene(args.reference/'scene.xml',args.output/'physics.xml')
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA not visible')
    factories=[functools.partial(make_environment,str(args.reference.resolve()),str(scene.resolve()),args.reward_mode,args.seed+i)
               for i in range(args.envs)]
    vector=SubprocVecEnv(factories,start_method='forkserver')
    logger=configure(str(args.output),['stdout','csv'])
    if args.resume:policy=PPO.load(args.resume,env=vector,device=args.device)
    else:policy=PPO('MlpPolicy',vector,device=args.device,n_steps=256,batch_size=min(1024,256*args.envs),
        n_epochs=5,learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.001,
        policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256]),log_std_init=-1.),seed=args.seed,verbose=1)
    policy.set_logger(logger)
    metadata=dict(schema='local_exploratory_contact_ppo_v1',reference=str(args.reference.resolve()),
        source_trajectory_sha256=hashlib.sha256((args.reference/'trajectory.npz').read_bytes()).hexdigest(),
        env_code_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'prism_reconstruction/contact_tracking_env.py').read_bytes()).hexdigest(),
        reward_mode=args.reward_mode,seed=args.seed,requested_steps=args.steps,environments=args.envs,device=args.device,
        algorithm='stable-baselines3 PPO 2.4.1',physics='MuJoCo 3.2.7, 500 Hz, control 50 Hz',
        training_resets='Uniform reference phase, 2-second episodes, no rollout root/object assistance',
        object_mass_kg=1.,mass_measured=False,domain_randomization=False,
        limitations=['Single reconstructed video and an imperfect geometric reference.',
            'Training returns do not establish full-clip success or sim2real transfer.',
            'Privileged object state is observed; no vision policy is trained here.',
            'This is an independent MuJoCo baseline, not an official PRISM training reproduction.'],
        safe_for_robot_execution=False)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    started=time.time()
    try:
        policy.learn(total_timesteps=args.steps,reset_num_timesteps=not bool(args.resume),
            callback=CheckpointCallback(save_freq=max(1,100000//args.envs),save_path=str(args.output/'checkpoints'),name_prefix='policy'))
        policy.save(args.output/'policy.zip')
        metadata.update(actual_steps=int(policy.num_timesteps),elapsed_s=time.time()-started,complete=True)
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    finally:vector.close()


if __name__=='__main__':main()
