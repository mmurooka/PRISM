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
import math

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.robot_physics import prepare_physics_scene


def make_environment(reference,scene,mode,seed,action_mode,horizon,teacher,residual_scale,contact_reward):
    from prism_reconstruction.contact_tracking_env import ContactTrackingEnv
    from stable_baselines3.common.monitor import Monitor
    env=Monitor(ContactTrackingEnv(reference,scene,reward_mode=mode,horizon_s=horizon,random_start=True,
        action_mode=action_mode,teacher_checkpoint=teacher,teacher_residual_scale=residual_scale,contact_reward=contact_reward))
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
    parser.add_argument('--action-mode',choices=('joint_offset','torque_fraction'),default='torque_fraction')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--horizon',type=float,default=2.)
    parser.add_argument('--teacher-checkpoint',type=Path,help='Optional released PRISM actor under the residual policy')
    parser.add_argument('--teacher-residual-scale',type=float,default=1.)
    parser.add_argument('--contact-reward',choices=('additive','coupled'),default='coupled')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<1 or args.envs<1:raise ValueError('Positive training budget and environment count required')
    if not math.isfinite(args.teacher_residual_scale) or args.teacher_residual_scale<=0:raise ValueError('Positive residual scale required')
    args.output.mkdir(parents=True)
    scene=prepare_physics_scene(args.reference/'scene.xml',args.output/'physics.xml')
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA not visible')
    teacher=str(args.teacher_checkpoint.resolve()) if args.teacher_checkpoint else None
    factories=[functools.partial(make_environment,str(args.reference.resolve()),str(scene.resolve()),args.reward_mode,args.seed+i,args.action_mode,args.horizon,teacher,args.teacher_residual_scale,args.contact_reward)
               for i in range(args.envs)]
    vector=SubprocVecEnv(factories,start_method='forkserver')
    logger=configure(str(args.output),['stdout','csv'])
    if args.resume:
        source_folder=args.resume.parent
        if source_folder.name=='checkpoints':source_folder=source_folder.parent
        source_meta=json.loads((source_folder/'metadata.json').read_text())
        if source_meta.get('action_mode','joint_offset')!=args.action_mode:raise ValueError('Cannot resume with different action semantics')
        if source_meta.get('teacher_checkpoint')!=teacher:raise ValueError('Cannot resume with a different teacher')
        if source_meta.get('teacher_residual_scale',1.)!=args.teacher_residual_scale:raise ValueError('Cannot resume with a different residual scale')
        policy=PPO.load(args.resume,env=vector,device=args.device,learning_rate=args.learning_rate)
    else:policy=PPO('MlpPolicy',vector,device=args.device,n_steps=256,batch_size=min(1024,256*args.envs),
        n_epochs=5,learning_rate=args.learning_rate,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.001,target_kl=.02,
        policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256]),
            log_std_init=-1.-math.log(args.teacher_residual_scale) if teacher else -1.),seed=args.seed,verbose=1)
    if teacher and not args.resume:
        # Begin at the validated teacher's deterministic action. Exploration is
        # still sampled from PPO's distribution during training.
        with torch.no_grad():
            policy.policy.action_net.weight.zero_()
            policy.policy.action_net.bias.zero_()
    policy.set_logger(logger)
    metadata=dict(schema='local_exploratory_contact_ppo_v1',reference=str(args.reference.resolve()),
        source_trajectory_sha256=hashlib.sha256((args.reference/'trajectory.npz').read_bytes()).hexdigest(),
        env_code_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'prism_reconstruction/contact_tracking_env.py').read_bytes()).hexdigest(),
        reward_mode=args.reward_mode,seed=args.seed,requested_steps=args.steps,environments=args.envs,device=args.device,
        action_mode=args.action_mode,learning_rate=args.learning_rate,horizon_s=args.horizon,
        teacher_checkpoint=teacher,
        teacher_residual_scale=args.teacher_residual_scale,
        contact_reward=args.contact_reward,
        teacher_checkpoint_sha256=hashlib.sha256(args.teacher_checkpoint.read_bytes()).hexdigest() if teacher else None,
        initial_residual_mean='zero' if teacher and not args.resume else 'default_or_resumed',
        initialization_checkpoint=str(args.resume.resolve()) if args.resume else None,
        initialization_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
        initial_timesteps=int(policy.num_timesteps),
        reward_changed_on_resume=bool(args.resume and source_meta.get('contact_reward','additive')!=args.contact_reward),
        code_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ('prism_reconstruction/contact_tracking_env.py','prism_reconstruction/robot_physics.py',
                         'prism_reconstruction/released_teacher.py','scripts/train_local_contact_policy.py')},
        algorithm='stable-baselines3 PPO 2.4.1',physics='MuJoCo 3.2.7, 500 Hz, control 50 Hz',
        training_resets='Uniform reference phase, configured episode horizon, no rollout root/object assistance',
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
        metadata.update(actual_steps=int(policy.num_timesteps),new_steps=int(policy.num_timesteps)-metadata['initial_timesteps'],
                        elapsed_s=time.time()-started,complete=True)
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    finally:vector.close()


if __name__=='__main__':main()
