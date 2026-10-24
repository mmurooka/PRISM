"""Differentiable unilateral force proposals for geometric trajectory refinement.

These are necessary centroidal balance conditions, not a dynamics controller:
individual joint effort, exact collision manifolds and no-slip constraints are
checked separately or remain limitations. Force-gap products discourage remote
support without turning uncertain contact hypotheses into observations.
"""
import math
import torch


def friction_rays(normals, coefficient, rays=8):
    """Inscribed Coulomb rays directed into the receiving body; unit normal load."""
    normals=normals/torch.linalg.vector_norm(normals,dim=-1,keepdim=True).clamp_min(1e-12)
    seed=torch.zeros_like(normals)
    axis=normals.abs().argmin(-1)
    seed.scatter_(-1,axis[...,None],1.)
    first=torch.cross(normals,seed,dim=-1)
    first=first/torch.linalg.vector_norm(first,dim=-1,keepdim=True)
    second=torch.cross(normals,first,dim=-1)
    theta=torch.arange(rays,device=normals.device,dtype=normals.dtype)*(2*math.pi/rays)
    return normals[...,None,:]+coefficient*(theta.cos()[...,None]*first[...,None,:]+theta.sin()[...,None]*second[...,None,:])


def wrench(points,forces):
    return forces.sum(-2),torch.cross(points,forces,dim=-1).sum(-2)


def frame_sum(values,frames,count):
    return torch.zeros((count,*values.shape[1:]),device=values.device,dtype=values.dtype).index_add(0,frames,values)


def normalized_wrench_error(actual,required,weight,length=.3,center=None):
    force=actual[0]-required[0];moment=actual[1]-required[1]
    if center is not None:moment=moment-torch.cross(center,force,dim=-1)
    return (force/weight).square().mean()+(moment/(weight*length)).square().mean()


class ContactWrenchProposal(torch.nn.Module):
    def __init__(self,frames,count,object_normals,stance,on_floor,box_half,robot_mass,box_mass,device,dtype):
        super().__init__()
        self.frames=frames;self.count=count;self.robot_weight=float(robot_mass)*9.81;self.box_weight=float(box_mass)*9.81
        self.object_rays=friction_rays(-object_normals,.6)
        normal=torch.tensor([0.,0.,1.],device=device,dtype=dtype)
        self.ground_rays=friction_rays(normal,.8)
        self.stance=stance[:,:,None,None].to(dtype)
        self.on_floor=on_floor[:,None,None].to(dtype)
        self.box_vertices=torch.cartesian_prod(*[torch.tensor([-1.,1.],device=device,dtype=dtype)]*3)*box_half
        self.robot_object=torch.nn.Parameter(torch.zeros((len(frames),8),device=device,dtype=dtype))
        active=stance.sum(1).clamp_min(1).to(dtype)
        initial=torch.ones((count,2,4,8),device=device,dtype=dtype)/(active[:,None,None,None]*32)
        self.feet=torch.nn.Parameter(initial)
        self.box_ground=torch.nn.Parameter(torch.zeros((count,8,8),device=device,dtype=dtype))

    def project(self):
        with torch.no_grad():
            for parameter in self.parameters():parameter.clamp_(min=0.)

    def forward(self,robot_points,object_points,box,box_rotation,feet,required_robot,required_box,robot_com=None):
        cf=self.frames
        local=torch.einsum('nr,nrj->nj',self.robot_object,self.object_rays)*self.box_weight
        object_forces=torch.einsum('nij,nj->ni',box_rotation[cf],local)
        foot_forces=torch.einsum('tsvr,rj->tsvj',self.feet*self.stance,self.ground_rays)*self.robot_weight
        ground_points=torch.einsum('tij,vj->tvi',box_rotation,self.box_vertices)+box[:,None]
        ground_forces=torch.einsum('tvr,rj->tvj',self.box_ground*self.on_floor,self.ground_rays)*self.box_weight
        robot_wrench=wrench(feet.flatten(1,2),foot_forces.flatten(1,2))
        robot_wrench=(robot_wrench[0]-frame_sum(object_forces,cf,self.count),
            robot_wrench[1]-frame_sum(torch.cross(robot_points,object_forces,dim=-1),cf,self.count))
        box_wrench=wrench(ground_points,ground_forces)
        box_wrench=(box_wrench[0]+frame_sum(object_forces,cf,self.count),
            box_wrench[1]+frame_sum(torch.cross(object_points,object_forces,dim=-1),cf,self.count))
        balance=normalized_wrench_error(robot_wrench,required_robot,self.robot_weight,center=robot_com)
        balance+=normalized_wrench_error(box_wrench,required_box,self.box_weight,center=box)
        load=self.robot_object.sum(-1)
        gap=(robot_points-object_points).square().sum(-1)
        complementarity=(load.square()*gap/.005**2).mean()
        complementarity+=((self.box_ground*self.on_floor).sum(-1).square()*ground_points[:,:,2].square()/.005**2).mean()
        complementarity+=((self.feet*self.stance).sum(-1).square()*feet[:,:,:,2].square()/.005**2).mean()
        effort=self.robot_object.square().mean()+self.feet.square().mean()+self.box_ground.square().mean()
        return balance+complementarity+.001*effort,dict(balance=balance,force_gap=complementarity,
            object_forces=object_forces,foot_forces=foot_forces,box_ground_forces=ground_forces)
