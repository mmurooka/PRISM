#!/usr/bin/env python3
"""Compare the adapter with pinned HoloSoma actor and observation functions.

Pure observation/rotation functions are extracted without JIT decorators to avoid
importing simulator backends. Their function bodies are not rewritten. This tests
feature conventions, not equality of the MuJoCo and Isaac Sim dynamics.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_tracking_env import ContactTrackingEnv


def source_functions(paths,requested,namespace):
    functions={}
    for path in paths:
        for node in ast.parse(Path(path).read_text()).body:
            if isinstance(node,ast.FunctionDef):functions[node.name]=node
    needed=set(requested);pending=list(requested)
    while pending:
        name=pending.pop()
        if name not in functions:continue
        for node in ast.walk(functions[name]):
            if isinstance(node,ast.Name) and node.id in functions and node.id not in needed:
                needed.add(node.id);pending.append(node.id)
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for name in sorted(needed):
        if name not in functions:continue
        node=functions[name];node.decorator_list=[];body.append(node)
    module=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
    exec(compile(module,'<pinned pure functions>','exec'),namespace)
    return namespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','reference','physics-scene','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    source=args.runtime_root/'src/holosoma/holosoma'
    sys.path.insert(0,str(source.parent))
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module
    from holosoma.config_types.algo import ModuleConfig,LayerConfig
    env=ContactTrackingEnv(args.reference,args.physics_scene,teacher_checkpoint=args.checkpoint)
    teacher=env.teacher;payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=dict(payload['experiment_config']['algo']['config']['module_dict']['actor'])
    config['layer_config']=LayerConfig(**config['layer_config'])
    upstream=setup_ppo_actor_module({'actor_obs':178},ModuleConfig(**config),29,.01,'cpu',{'actor_obs':1})
    upstream.load_state_dict(payload['actor_model_state_dict'],strict=True);upstream.eval()
    torch.manual_seed(7);random=torch.randn(100,178)
    with torch.no_grad():a=upstream.act_inference({'actor_obs':random});b=teacher.actor(random)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    class MockMotion:pass
    paths=[source/'utils/torch_utils.py',source/'utils/rotations.py',source/'managers/observation/terms/wbt.py']
    namespace=dict(torch=torch,np=np,Tensor=torch.Tensor,torch_nn_func=torch.nn.functional,
        MotionCommand=MockMotion)
    original=source_functions(paths,list(teacher.terms),namespace)
    def tensor(value):return torch.as_tensor(np.asarray(value),dtype=torch.float32).unsqueeze(0)
    def quaternion(matrix):return tensor(Rotation.from_matrix(matrix).as_quat())
    differences=[];action_differences=[];rng=np.random.default_rng(71)
    for phase in np.linspace(0,env.times[-1]-.1,20):
        env.reset(options={'start_time':float(phase)})
        velocity=rng.normal(0,.05,env.model.nv);mujoco.mj_integratePos(env.model,env.data.qpos,velocity,1.)
        env.data.qvel[:]=rng.normal(0,.3,env.model.nv);mujoco.mj_forward(env.model,env.data)
        teacher.previous_action=rng.normal(0,.3,29)
        actual=teacher.observation(env);q,v,_=env._reference(phase)
        ref=teacher.reference_data;data=env.data;body=teacher.ref_body;box=env.box
        R=data.xmat[body].reshape(3,3);root=data.xmat[1].reshape(3,3);chart=teacher.object_chart
        motion=MockMotion();motion.motion=SimpleNamespace(has_object=True)
        motion.command=tensor(np.r_[q[env.qadr],v[env.vadr]])
        motion.robot_ref_pos_w=tensor(data.xpos[body]);motion.robot_ref_quat_w=quaternion(R)
        motion.ref_pos_w=tensor(ref.xpos[body]);motion.ref_quat_w=quaternion(ref.xmat[body].reshape(3,3))
        motion.simulator_object_pos_w=tensor(data.xpos[box]);motion.simulator_object_quat_w=quaternion(data.xmat[box].reshape(3,3)@chart)
        motion.object_pos_w=tensor(ref.xpos[box]);motion.object_quat_w=quaternion(ref.xmat[box].reshape(3,3)@chart)
        motion.object_size=tensor(2*env.model.geom_size[env.box_geom][[0,2,1]])
        state=np.r_[data.qpos[:3],Rotation.from_matrix(root).as_quat(),data.qvel[:3],root@data.qvel[3:6]]
        mock=SimpleNamespace(num_envs=1,base_quat=quaternion(root),default_dof_pos=tensor(teacher.default),
            simulator=SimpleNamespace(robot_root_states=tensor(state),dof_pos=tensor(data.qpos[env.qadr]),dof_vel=tensor(data.qvel[env.vadr])),
            action_manager=SimpleNamespace(action=tensor(teacher.previous_action)),
            command_manager=SimpleNamespace(get_state=lambda name:motion))
        pieces=[original[name](mock)*teacher.terms[name].get('scale',1.) for name in sorted(teacher.terms)]
        expected=torch.cat(pieces,dim=-1).clamp(-100,100)
        np.testing.assert_allclose(actual,expected[0].numpy(),atol=2e-6,rtol=2e-6)
        differences.append(float(np.max(np.abs(actual-expected[0].numpy()))))
        with torch.no_grad():
            first=upstream.act_inference({'actor_obs':expected});second=teacher.actor(torch.from_numpy(actual[None]))
            action_differences.append(float((first-second).abs().max()))
    result=dict(checkpoint_sha256=teacher.sha256,random_actor_samples=100,actor_difference_identical_inputs=0.,
        observation_states=20,maximum_observation_difference=max(differences),
        maximum_action_difference_from_observation_roundoff=max(action_differences),
        source_sha256={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        interpretation='Pure function and actor parity only; simulator/contact geometry transfer remains unverified by this test.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
