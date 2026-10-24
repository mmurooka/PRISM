"""Differentiable FK for the two-free-root, 29-hinge G1/object MuJoCo scene."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import mujoco


def rotation_6d(value):
    first=torch.nn.functional.normalize(value[...,:3],dim=-1)
    second=value[...,3:]-first*(first*value[...,3:]).sum(-1,keepdim=True)
    second=torch.nn.functional.normalize(second,dim=-1)
    return torch.stack((first,second,torch.cross(first,second,dim=-1)),dim=-1)


def matrix_6d(value):return torch.cat((value[...,0],value[...,1]),dim=-1)


class TorchRobot:
    def __init__(self,model,device='cpu',dtype=torch.float64):
        self.model=model;self.device=device;self.dtype=dtype
        self.box=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'hoi_box')
        self.hinges=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        self.index={int(j):i for i,j in enumerate(self.hinges)}
        self.position=self.tensor(model.body_pos)
        self.rotation=self.tensor(Rotation.from_quat(model.body_quat[:,[1,2,3,0]]).as_matrix())
        self.anchor=self.tensor(model.jnt_pos);self.axis=self.tensor(model.jnt_axis)
        self.parent=model.body_parentid.copy()
        self.joints=[]
        for body in range(model.nbody):
            joints=list(range(model.body_jntadr[body],model.body_jntadr[body]+model.body_jntnum[body]))
            if body in (0,1,self.box):self.joints.append(None);continue
            if len(joints)>1 or any(model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE for j in joints):
                raise ValueError('Only zero/one hinge per non-root link is supported')
            self.joints.append(joints[0] if joints else None)

    def tensor(self,value):return torch.as_tensor(value,dtype=self.dtype,device=self.device)

    def forward(self,joints,root_position,root_rotation,box_position,box_rotation):
        count=len(joints);identity=torch.eye(3,dtype=self.dtype,device=self.device)
        positions=[torch.zeros((count,3),dtype=self.dtype,device=self.device)]
        rotations=[identity.expand(count,-1,-1)]
        for body in range(1,self.model.nbody):
            if body in (1,self.box):
                positions.append(root_position if body==1 else box_position)
                rotations.append(root_rotation if body==1 else box_rotation);continue
            parent=int(self.parent[body]);local_rotation=self.rotation[body].expand(count,-1,-1)
            local_position=self.position[body].expand(count,-1);joint=self.joints[body]
            if joint is not None:
                axis=self.axis[joint];x,y,z=axis.unbind();zero=x*0
                skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
                angle=joints[:,self.index[joint]][:,None,None]
                turn=identity+torch.sin(angle)*skew+(1-torch.cos(angle))*(skew@skew)
                anchor=self.anchor[joint]
                local_position=local_position+torch.einsum('bij,bj->bi',local_rotation,
                    anchor-torch.einsum('bij,j->bi',turn,anchor))
                local_rotation=local_rotation@turn
            positions.append(positions[parent]+torch.einsum('bij,bj->bi',rotations[parent],local_position))
            rotations.append(rotations[parent]@local_rotation)
        return torch.stack(positions,dim=1),torch.stack(rotations,dim=1)


def acceleration(values,dt):
    interior=(values[2:]-2*values[1:-1]+values[:-2])/dt**2
    return torch.cat((interior[:1],interior,interior[-1:]),dim=0)


def velocity(values,dt):
    return torch.cat(((values[1:2]-values[:1])/dt,
        (values[2:]-values[:-2])/(2*dt),(values[-1:]-values[-2:-1])/dt),dim=0)


def centroidal_contact_wrench(positions,rotations,masses,inertia,dt):
    """Required external contact wrench about the fixed world origin.

    Positions are body COMs; rotations map principal inertia axes to world.
    This includes angular momentum and vertical acceleration, but does not
    certify individual contact forces or motor torques. Boundary derivatives
    have lower accuracy and must be reported as such.
    """
    linear_velocity=velocity(positions,dt)
    angular_matrix=velocity(rotations,dt)@rotations.transpose(-1,-2)
    omega=.5*torch.stack((angular_matrix[...,2,1]-angular_matrix[...,1,2],
        angular_matrix[...,0,2]-angular_matrix[...,2,0],
        angular_matrix[...,1,0]-angular_matrix[...,0,1]),dim=-1)
    world_inertia=(rotations*inertia[None,:,None,:])@rotations.transpose(-1,-2)
    angular_momentum=torch.cross(positions,masses[None,:,None]*linear_velocity,dim=-1)
    angular_momentum+=(world_inertia@omega[...,None]).squeeze(-1)
    gravity=torch.zeros_like(positions);gravity[...,2]=-9.81*masses[None,:]
    force=(masses[None,:,None]*acceleration(positions,dt)-gravity).sum(1)
    moment=velocity(angular_momentum.sum(1),dt)-torch.cross(positions,gravity,dim=-1).sum(1)
    return force,moment
