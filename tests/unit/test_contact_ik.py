"""Analytic Jacobian regression tests independent of proprietary body assets."""
import sys
from pathlib import Path
import unittest
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.contact_ik import point_jacobian,distance_jacobian,relaxed_contact_error


class ContactIKTests(unittest.TestCase):
    def test_tangential_relaxation_keeps_normal_error_and_correct_gradient(self):
        for difference in (np.array([.03,.02,.01]),np.array([.15,.2,.04])):
            error,jac=relaxed_contact_error(difference,np.array([0.,0.,1.]))
            self.assertAlmostEqual(error[2],difference[2])
            for i in range(3):
                shifted=difference.copy();shifted[i]+=1e-6
                after,_=relaxed_contact_error(shifted,np.array([0.,0.,1.]))
                np.testing.assert_allclose((after-error)/1e-6,jac[:,i],atol=3e-6)
    def test_point_jacobian_uses_quaternion_tangent_coordinates(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="box" size=".2 .3 .4"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        q=data.qpos.copy();local=np.array([.12,-.2,.3])
        point,jac=point_jacobian(model,data,1,local)
        for column in range(model.nv):
            step=np.zeros(model.nv);step[column]=1e-6;changed=q.copy()
            mujoco.mj_integratePos(model,changed,step,1)
            data.qpos[:]=changed;mujoco.mj_forward(model,data)
            actual,_=point_jacobian(model,data,1,local)
            np.testing.assert_allclose((actual-point)/1e-6,jac[:,column],atol=2e-6)

    def test_signed_distance_gradient_inside_and_outside_collision(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/><body><freejoint/><geom type="sphere" size=".2"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model)
        for height in (.18,.25):
            data.qpos[2]=height;mujoco.mj_forward(model,data)
            value,jac=distance_jacobian(model,data,1,0,.1)
            self.assertAlmostEqual(value,height-.2)
            data.qpos[2]+=1e-6;mujoco.mj_forward(model,data)
            shifted,_=distance_jacobian(model,data,1,0,.1)
            self.assertAlmostEqual((shifted-value)/1e-6,jac[2],places=6)
            self.assertAlmostEqual(jac[2],1.)


if __name__=='__main__':unittest.main()
