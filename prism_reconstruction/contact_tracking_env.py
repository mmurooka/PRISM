"""Experimental contact-aware residual tracking in local MuJoCo.

Only the 29 joint motors act on the robot. The floating box receives gravity and
collision forces; there are no object springs, welds, teleports during a rollout,
or root assistance. Resetting a sampled reference state is a training reset.
"""
from pathlib import Path
import json

import gymnasium as gym
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .robot_physics import motor_parameters,reference_velocities


REGIONS=('left_hand','right_hand','left_forearm','right_forearm','left_upper_arm','right_upper_arm',
         'left_thigh','right_thigh','left_knee','right_knee','left_shin','right_shin',
         'left_foot','right_foot','pelvis','torso','head','other')


class ContactTrackingEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,reference,physics_scene,reward_mode='whole_body',horizon_s=2.,random_start=True,
                 randomize=False):
        if reward_mode not in ('whole_body','hands_only'):raise ValueError('Unknown reward variant')
        self.reference=Path(reference);self.reward_mode=reward_mode;self.horizon_s=float(horizon_s)
        self.random_start=random_start;self.randomize=randomize
        self.model=m=mujoco.MjModel.from_xml_path(str(physics_scene));self.data=d=mujoco.MjData(m)
        self.motor=motor_parameters(m);self.qadr=self.motor['qadr'];self.vadr=self.motor['vadr']
        with np.load(self.reference/'trajectory.npz') as a:
            self.qref=a['qpos'].copy();self.times=a['timestamps_s'].copy();self.stance=a['stance'].copy()
        if self.qref.shape!=(len(self.times),m.nq):raise ValueError('Reference/model configuration mismatch')
        self.records=json.loads((self.reference/'diagnostics.json').read_text())
        self.meta=json.loads((self.reference/'metadata.json').read_text())
        if self.meta['independent_diagnostic_frames']:raise ValueError('A continuous trajectory is required')
        self.vref=gaussian_filter1d(reference_velocities(m,self.qref,self.times),1.,axis=0)
        self.box=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'hoi_box')
        self.box_joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hoi_box_free')
        self.box_q=int(m.jnt_qposadr[self.box_joint]);self.box_v=int(m.jnt_dofadr[self.box_joint])
        self.box_geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'hoi_box_geom')
        self.base_mass=m.body_mass[self.box];self.base_inertia=m.body_inertia[self.box].copy()
        self.friction=m.geom_friction.copy();self.control_dt=.02
        self.substeps=int(round(self.control_dt/m.opt.timestep))
        if not np.isclose(self.substeps*m.opt.timestep,self.control_dt):raise ValueError('Control/physics steps must divide exactly')
        self.action_space=gym.spaces.Box(-1.,1.,(29,),np.float32)
        self.previous_action=np.zeros(29);self.elapsed=0.;self.start_time=0.
        d.qpos[:]=self.qref[0];mujoco.mj_forward(m,d)
        observation=self._observation()
        self.observation_space=gym.spaces.Box(-100.,100.,observation.shape,np.float32)

    def _reference(self,time):
        time=float(np.clip(time,self.times[0],self.times[-1]));right=int(np.searchsorted(self.times,time,side='right'))
        right=min(max(1,right),len(self.times)-1);left=right-1
        alpha=(time-self.times[left])/(self.times[right]-self.times[left])
        q=(1-alpha)*self.qref[left]+alpha*self.qref[right]
        for start in (3,self.box_q+3):
            first=self.qref[left,start:start+4];last=self.qref[right,start:start+4]
            if first@last<0:last=-last
            quat=(1-alpha)*first+alpha*last;q[start:start+4]=quat/np.linalg.norm(quat)
        v=(1-alpha)*self.vref[left]+alpha*self.vref[right]
        return q,v,left if alpha<.5 else right

    def _targets(self,index):
        return [target for target in self.records[index]['targets'] if self.reward_mode=='whole_body' or
                target['region'] in ('left_hand','right_hand')]

    def _contact_positions(self,target):
        d=self.data;body=target['robot_body_id']
        robot=d.xpos[body]+d.xmat[body].reshape(3,3)@target['robot_point_local']
        box=d.xpos[self.box]+d.xmat[self.box].reshape(3,3)@target['point_object']
        return robot,box

    def _observation(self):
        m,d=self.model,self.data;phase=self.start_time+self.elapsed
        q,v,index=self._reference(phase);delta=np.zeros(m.nv)
        mujoco.mj_differentiatePos(m,delta,1.,d.qpos,q)
        R=d.xmat[1].reshape(3,3);object_R=d.xmat[self.box].reshape(3,3)
        slots=np.zeros((len(REGIONS),4))
        # Reward ablations retain the same observations and allowed contacts.
        for target in self.records[index]['targets']:
            if target['region'] not in REGIONS:continue
            robot,box=self._contact_positions(target);i=REGIONS.index(target['region'])
            slots[i,:3]=(box-robot)@R;slots[i,3]=target['weight']
        future=np.concatenate([self._reference(phase+offset)[0][self.qadr]-d.qpos[self.qadr] for offset in (.1,.3,.5)])
        obs=np.concatenate([d.qpos[self.qadr],d.qvel[self.vadr]*.1,self.previous_action,
            q[self.qadr]-d.qpos[self.qadr],v[self.vadr]*.1,delta[:6],delta[self.box_v:],
            R.T@np.array([0.,0.,-1.]),d.qvel[:6]*.1,(d.xpos[self.box]-d.xpos[1])@R,
            (R.T@object_R)[:,:2].reshape(-1),slots.reshape(-1),future,self.stance[index].astype(float),
            [np.sin(2*np.pi*phase/self.times[-1]),np.cos(2*np.pi*phase/self.times[-1])]])
        return np.clip(obs,-100,100).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {};m,d=self.model,self.data
        upper=max(self.times[0],self.times[-1]-.25)
        self.start_time=float(options.get('start_time',self.np_random.uniform(self.times[0],upper) if self.random_start else self.times[0]))
        if not self.times[0]<=self.start_time<self.times[-1]:raise ValueError('Reset phase outside reference')
        self.elapsed=0.;self.previous_action=np.zeros(29)
        q,v,_=self._reference(self.start_time);mujoco.mj_resetData(m,d)
        factor=self.np_random.uniform(.7,1.3) if self.randomize else 1.
        m.body_mass[self.box]=self.base_mass*factor;m.body_inertia[self.box]=self.base_inertia*factor
        m.geom_friction[:]=self.friction
        if self.randomize:m.geom_friction[:,0]*=self.np_random.uniform(.8,1.2)
        mujoco.mj_setConst(m,d)
        d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
        return self._observation(),dict(start_time_s=self.start_time,box_mass_kg=float(m.body_mass[self.box]))

    def step(self,action):
        m,d=self.model,self.data;action=np.asarray(action,float)
        if action.shape!=(29,) or not np.isfinite(action).all():raise ValueError('Finite 29-joint action required')
        action=np.clip(action,-1,1);q,_,_=self._reference(self.start_time+self.elapsed)
        desired=q[self.qadr]+.25*action
        for _ in range(self.substeps):
            torque=self.motor['kp']*(desired-d.qpos[self.qadr])-self.motor['kd']*d.qvel[self.vadr]
            d.ctrl[:]=np.clip(torque,self.motor['lower'],self.motor['upper'])
            mujoco.mj_step(m,d)
        self.elapsed+=self.control_dt;mujoco.mj_forward(m,d)
        q,_,index=self._reference(self.start_time+self.elapsed)
        difference=np.zeros(m.nv);mujoco.mj_differentiatePos(m,difference,1.,d.qpos,q)
        root_error=float(np.linalg.norm(difference[:3]));object_error=float(np.linalg.norm(difference[self.box_v:self.box_v+3]))
        posture=np.exp(-np.mean(difference[self.vadr]**2)/.2**2)
        root=np.exp(-root_error**2/.15**2-np.linalg.norm(difference[3:6])**2/.5**2)
        object_tracking=np.exp(-object_error**2/.12**2-np.linalg.norm(difference[self.box_v+3:])**2/.5**2)
        forces={};unwanted=0
        for i in range(d.ncon):
            c=d.contact[i];first=int(m.geom_bodyid[c.geom1]);second=int(m.geom_bodyid[c.geom2])
            if self.box not in (first,second):continue
            robot=second if first==self.box else first
            if robot==0:continue
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
            weld=int(m.body_weldid[robot]);forces[weld]=forces.get(weld,0.)+max(0.,float(force[0]))
        targets=self._targets(index);weights=[];errors=[];supported=[]
        allowed={int(m.body_weldid[target['robot_body_id']]) for target in self.records[index]['targets']}
        for target in targets:
            robot,box=self._contact_positions(target);errors.append(float(np.linalg.norm(robot-box)))
            weights.append(target['weight']);weld=int(m.body_weldid[target['robot_body_id']])
            supported.append(forces.get(weld,0.)>1.)
        unwanted=sum(force>1 and weld not in allowed for weld,force in forces.items())
        contact_position=float(np.average(np.exp(-np.square(errors)/.03**2),weights=weights)) if weights else 1.
        contact_force=float(np.average(supported,weights=weights)) if weights else 1.
        effort=float(np.mean((d.ctrl/np.maximum(self.motor['upper'],1))**2))
        action_rate=float(np.mean((action-self.previous_action)**2))
        reward=.25*posture+.2*root+.3*object_tracking+.15*contact_position+.1*contact_force+.1
        reward-=.01*effort+.01*action_rate+.03*unwanted
        tilt=float(d.xmat[1].reshape(3,3)[2,2])
        finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        warnings=int(sum(warning.number for warning in d.warning))
        terminated=not finite or warnings>0 or d.qpos[2]<.3 or tilt<.25 or root_error>.5 or object_error>.6
        truncated=self.elapsed>=self.horizon_s or self.start_time+self.elapsed>=self.times[-1]
        if terminated:reward-=1.
        self.previous_action=action
        info=dict(root_error_m=root_error,object_error_m=object_error,contact_position_reward=contact_position,
            contact_force_fraction=contact_force,contact_error_mean_m=float(np.mean(errors)) if errors else 0.,
            active_contact_targets=len(targets),unwanted_object_contact_bodies=int(unwanted),
            reference_time_s=float(self.start_time+self.elapsed),box_height_m=float(d.xpos[self.box,2]),
            physics_warning_count=warnings,
            terminated_by_tracking_failure=bool(terminated))
        return self._observation(),float(reward),bool(terminated),bool(truncated),info
