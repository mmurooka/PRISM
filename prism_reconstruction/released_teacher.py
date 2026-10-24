"""Strict adapter for PRISM's released 178-input privileged teacher.

Observation terms follow the pinned HoloSoma implementation, including its
alphabetical concatenation and torso-relative object/reference observations.
This adapter changes simulator backend and object geometry; transfer is evaluated,
not assumed. It does not claim to reproduce the Isaac Sim training environment.
"""
from pathlib import Path
import hashlib

import mujoco
import numpy as np
import torch


class ReleasedTeacher:
    def __init__(self,checkpoint,model,motor):
        self.path=Path(checkpoint);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        payload=torch.load(self.path,map_location='cpu',weights_only=False)
        config=payload['experiment_config'];actor=config['algo']['config']['module_dict']['actor']
        layers=actor['layer_config'];self.terms=config['observation']['groups']['actor_obs']['terms']
        if actor['type']!='MLP' or layers['activation']!='ELU' or layers.get('use_layer_norm'):
            raise ValueError('Only the released plain ELU teacher MLP is supported')
        if config['observation']['groups']['actor_obs']['history_length']!=1:raise ValueError('Unexpected observation history')
        if any(payload.get('actor_obs_normalizer_state',{}).get('actor_obs',{})):
            raise ValueError('Nonempty teacher normalization needs an explicit adapter')
        expected={'motion_command','motion_ref_ori_b','base_ang_vel','base_lin_vel','dof_pos','dof_vel','actions',
                  'obj_pos_b','obj_ori_b','obj_target_pos_b','obj_target_ori_b','obj_size'}
        if set(self.terms)!=expected:raise ValueError('Unsupported teacher observation vocabulary')
        for name,term in self.terms.items():
            if term['func']!='holosoma.managers.observation.terms.wbt:'+name or term.get('params'):
                raise ValueError(f'Unexpected observation semantics: {name}')
        self.names=config['robot']['dof_names']
        if self.names!=motor['names']:raise ValueError('Teacher/model joint order mismatch')
        default=config['robot']['init_state']['default_joint_angles']
        self.default=np.asarray([default[name] for name in self.names])
        control=config['robot']['control']
        if not control['action_scales_by_effort_limit_over_p_gain'] or control['control_type']!='P':
            raise ValueError('Unexpected teacher action semantics')
        def gain(name,table):
            matches=[value for key,value in table.items() if key in name]
            if len(matches)!=1:raise ValueError(f'Ambiguous motor gain {name}')
            return matches[0]
        self.kp=np.array([gain(name,control['stiffness']) for name in self.names])
        self.kd=np.array([gain(name,control['damping']) for name in self.names])
        self.action_scale=control['action_scale']*np.array(config['robot']['dof_effort_limit_list'])/self.kp
        self.clip=float(control['action_clip_value'])
        motion=config['command']['setup_terms']['motion_command']['params']['motion_config']
        if motion['num_future_steps']!=0 or motion['align_motion_to_init_yaw']:
            raise ValueError('Unsupported teacher command alignment')
        self.ref_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,motion['body_name_ref'][0])
        if self.ref_body<0:raise ValueError('Teacher reference link is missing')
        dimensions=[178,*layers['hidden_dims'],29];network=[]
        for i,(source,target) in enumerate(zip(dimensions[:-1],dimensions[1:])):
            network.append(torch.nn.Linear(source,target))
            if i<len(dimensions)-2:network.append(torch.nn.ELU())
        self.actor=torch.nn.Sequential(*network).eval()
        state={key.removeprefix('actor_module.module.'):value for key,value in payload['actor_model_state_dict'].items()
               if key.startswith('actor_module.module.')}
        self.actor.load_state_dict(state,strict=True);self.actor.requires_grad_(False)
        self.reference_data=mujoco.MjData(model);self.previous_action=np.zeros(29)
        # Our reconstructed cuboid uses (width,height,thickness). The released
        # box-policy chart uses (width,depth,height). This proper rotation keeps
        # the physical shape unchanged, and is applied to both object rotations.
        self.object_chart=np.array([[1.,0,0],[0,0,1],[0,-1,0]])

    def reset(self):self.previous_action[:]=0.

    def observation(self,env):
        model,data=env.model,env.data
        q,v,_=env._reference(env.start_time+env.elapsed)
        ref=self.reference_data;ref.qpos[:]=q;mujoco.mj_kinematics(model,ref)
        torso=data.xmat[self.ref_body].reshape(3,3);base=data.xmat[1].reshape(3,3)
        position=data.xpos[self.ref_body];box=env.box
        terms=dict(actions=self.previous_action.copy(),base_ang_vel=data.qvel[3:6].copy(),
            base_lin_vel=base.T@data.qvel[:3],dof_pos=data.qpos[env.qadr]-self.default,dof_vel=data.qvel[env.vadr].copy(),
            motion_command=np.r_[q[env.qadr],v[env.vadr]],
            motion_ref_ori_b=(torso.T@ref.xmat[self.ref_body].reshape(3,3))[:,:2].reshape(-1),
            obj_pos_b=torso.T@(data.xpos[box]-position),
            obj_ori_b=(torso.T@data.xmat[box].reshape(3,3)@self.object_chart)[:,:2].reshape(-1),
            obj_target_pos_b=torso.T@(ref.xpos[box]-position),
            obj_target_ori_b=(torso.T@ref.xmat[box].reshape(3,3)@self.object_chart)[:,:2].reshape(-1),
            obj_size=2*model.geom_size[env.box_geom][[0,2,1]])
        observations=[]
        for name in sorted(self.terms):
            value=terms[name]*self.terms[name].get('scale',1.)
            clip=self.terms[name].get('clip')
            if clip is not None:value=np.clip(value,*clip)
            observations.append(value)
        result=np.clip(np.concatenate(observations),-100,100).astype(np.float32)
        if result.shape!=(178,) or not np.isfinite(result).all():raise ValueError('Invalid teacher observation')
        return result

    @torch.no_grad()
    def desired_positions(self,env,residual):
        action=self.actor(torch.from_numpy(self.observation(env)[None]))[0].numpy()
        # The new policy changes bounded fractions of the same motor-normalized
        # action coordinates; it never changes the floating robot/object root.
        applied=np.clip(action+residual,-self.clip,self.clip)
        self.previous_action=applied.copy()
        return self.default+self.action_scale*applied
