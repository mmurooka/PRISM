#!/usr/bin/env python3
"""Validate expanded-actor initialization and equivalent control coordinates."""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium.utils.env_checker import check_env
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_tracking_env import ContactTrackingEnv
from prism_reconstruction.teacher_finetune import initialize_actor_from_teacher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','physics-scene','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    kwargs=dict(teacher_checkpoint=args.checkpoint,contact_reward='coupled')
    env=ContactTrackingEnv(args.reference,args.physics_scene,policy_control='teacher_absolute',**kwargs)
    base=ContactTrackingEnv(args.reference,args.physics_scene,**kwargs)
    policy=PPO('MlpPolicy',env,n_steps=16,batch_size=16,device='cpu',policy_kwargs=dict(
        net_arch=dict(pi=[512,256,128],vf=[512,256,128]),activation_fn=torch.nn.ELU,log_std_init=-1.))
    initialize_actor_from_teacher(policy.policy,env.teacher.actor);rng=np.random.default_rng(2);errors=[]
    for phase in np.linspace(0,env.times[-1]-.01,100):
        env.reset(options={'start_time':float(phase)});env.data.qpos[env.qadr]+=rng.normal(0,.01,29)
        mujoco.mj_forward(env.model,env.data);observation=env._observation()
        with torch.no_grad():expected=env.teacher.actor(torch.tensor(observation[:178][None]))[0].numpy()
        actual=policy.predict(observation,deterministic=True)[0]
        errors.append(float(abs(actual-np.clip(expected,-env.teacher.clip,env.teacher.clip)).max()))
    if max(errors)>1e-4:raise AssertionError('Expanded actor no longer preserves teacher output')
    state_errors=[];steps=0
    for phase in (0.,2.,4.,4.79,8.):
        env.reset(options={'start_time':phase});base.reset(options={'start_time':phase})
        for _ in range(20):
            action=policy.predict(env._observation(),deterministic=True)[0].astype(float)
            with torch.no_grad():baseline=base.teacher.actor(torch.tensor(base.teacher.observation(base)[None]))[0].numpy()
            residual=action-baseline.astype(float)
            if abs(residual).max()>1e-3:raise AssertionError('Equivalent-action correction unexpectedly large')
            _,_,done,_,_=env.step(action);base.step(residual)
            state_errors.append(float(abs(env.data.qpos-base.data.qpos).max()));steps+=1
            if done:break
    if max(state_errors)>1e-10:raise AssertionError('Equivalent commands changed physical transitions')
    check_env(env,skip_render_check=True)
    result=dict(teacher_checkpoint_sha256=env.teacher.sha256,sample_states=100,
        observation_dimensions=env.observation_space.shape[0],maximum_action_difference=max(errors),
        equivalent_command_steps=steps,maximum_qpos_difference_with_identical_commands=max(state_errors),gym_checks_passed=True,
        interpretation='Actor initialization and identical-command physics parity only. Independent closed-loop trajectories can diverge from floating-point contact sensitivity.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
