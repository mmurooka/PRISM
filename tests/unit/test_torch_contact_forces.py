import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.torch_contact_forces import friction_rays,wrench,frame_sum,ContactWrenchProposal,normalized_wrench_error,cuboid_edge_rays

class ContactForceTests(unittest.TestCase):
    def test_bottom_edge_can_propose_upward_support_without_side_pressure(self):
        points=torch.tensor([[1.,-1.,0.],[1.,0.,0.]],dtype=torch.float64)
        rays,count=cuboid_edge_rays(points,torch.ones(3,dtype=points.dtype),.6)
        torch.testing.assert_close(count,torch.tensor([2,1]))
        upward=rays[0,8:16].mean(0)
        torch.testing.assert_close(upward,torch.tensor([0.,1.,0.],dtype=points.dtype),atol=1e-14,rtol=0)
        self.assertEqual(float(rays[1,8:].abs().sum()),0.)

    def test_wrench_residual_is_invariant_to_world_origin(self):
        torch.manual_seed(19)
        actual=(torch.randn(5,3,dtype=torch.float64),torch.randn(5,3,dtype=torch.float64))
        required=(torch.randn(5,3,dtype=torch.float64),torch.randn(5,3,dtype=torch.float64))
        center=torch.randn(5,3,dtype=torch.float64);offset=torch.tensor([13.,-7.,21.],dtype=torch.float64).expand(5,3)
        before=normalized_wrench_error(actual,required,100.,center=center)
        moved_actual=(actual[0],actual[1]+torch.cross(offset,actual[0],dim=-1))
        moved_required=(required[0],required[1]+torch.cross(offset,required[0],dim=-1))
        after=normalized_wrench_error(moved_actual,moved_required,100.,center=center+offset)
        torch.testing.assert_close(before,after,rtol=1e-12,atol=1e-12)

    def test_static_floor_support_has_zero_wrench_error(self):
        dtype=torch.float64;cf=torch.arange(3);n=torch.tensor([[1.,0,0]]*3,dtype=dtype)
        module=ContactWrenchProposal(cf,3,n,torch.ones(3,2,dtype=torch.bool),torch.ones(3,dtype=dtype),torch.ones(3,dtype=dtype)*.2,10.,1.,'cpu',dtype)
        with torch.no_grad():module.box_ground[:,module.box_vertices[:,2]<0,:]=1/32
        points=torch.zeros(3,3,dtype=dtype);box=points.clone();box[:,2]=.2
        feet=torch.zeros(3,2,4,3,dtype=dtype);R=torch.eye(3,dtype=dtype).repeat(3,1,1)
        robot=(torch.tensor([[0.,0,98.1]]*3,dtype=dtype),points.clone())
        obj=(torch.tensor([[0.,0,9.81]]*3,dtype=dtype),points.clone())
        _,parts=module(points,points,box,R,feet,robot,obj)
        self.assertLess(float(parts['balance']),1e-25);self.assertLess(float(parts['force_gap']),1e-25)
        with torch.no_grad():module.box_ground.zero_()
        _,unsupported=module(points,points,box,R,feet,robot,obj)
        self.assertGreater(float(unsupported['balance']),.3)

    def test_rays_obey_unilateral_coulomb_cone(self):
        n=torch.tensor([[1.,0,0],[0,-1.,0],[.3,.4,.5]],dtype=torch.float64)
        n=n/n.norm(dim=-1,keepdim=True);r=friction_rays(n,.6)
        load=(r*n[:,None]).sum(-1)
        torch.testing.assert_close(load,torch.ones_like(load))
        torch.testing.assert_close((r-load[...,None]*n[:,None]).norm(dim=-1),torch.full_like(load,.6))
        coeff=torch.rand(3,8,dtype=n.dtype);force=(r*coeff[...,None]).sum(1)
        normal=(force*n).sum(-1)
        self.assertTrue(torch.all((force-normal[:,None]*n).norm(dim=-1)<=.6*normal+1e-10))

    def test_opposite_contacts_support_box_and_load_robot(self):
        dtype=torch.float64;cf=torch.tensor([0,0,1,1,2,2]);n=torch.tensor([[1.,0,0],[-1.,0,0]]*3,dtype=dtype)
        module=ContactWrenchProposal(cf,3,n,torch.ones(3,2,dtype=torch.bool),torch.zeros(3,dtype=dtype),torch.ones(3,dtype=dtype)*.2,10.,1.,'cpu',dtype)
        with torch.no_grad():module.robot_object.fill_(.03)
        # Equal side loads cancel horizontally; opposite friction tangents lift.
        force=torch.tensor([[0,0,4.905],[0,0,4.905]]*3,dtype=dtype)
        points=torch.tensor([[.2,0,1.],[-.2,0,1.]]*3,dtype=dtype)
        torch.testing.assert_close(frame_sum(force,cf,3),torch.tensor([[0,0,9.81]]*3,dtype=dtype))
        torch.testing.assert_close(frame_sum(torch.cross(points,force,dim=-1),cf,3),torch.zeros(3,3,dtype=dtype))
        # Exercise the complete differentiable objective, including force-gap coupling.
        feet=torch.zeros(3,2,4,3,dtype=dtype);R=torch.eye(3,dtype=dtype).repeat(3,1,1)
        required=(torch.zeros(3,3,dtype=dtype),torch.zeros(3,3,dtype=dtype))
        p=points.clone().requires_grad_();q=points.clone();q[:,2]+=.02
        value,parts=module(p,q,torch.zeros(3,3,dtype=dtype),R,feet,required,required)
        value.backward();self.assertTrue(torch.isfinite(value));self.assertGreater(p.grad.abs().sum(),0)
        self.assertGreater(parts['force_gap'],0)
        with torch.no_grad():module.robot_object[:]=-1
        module.project();self.assertTrue(torch.all(module.robot_object>=0))

if __name__=='__main__':unittest.main()
