import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.inverse_contact_audit import contact_matrix,fit_required_force


class InverseContactAuditTests(unittest.TestCase):
    def test_floor_force_sign_and_missing_support(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="2 2 .1" friction=".6 .01 .001"/>
          <body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1" mass="2"/></body>
        </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        matrix,contacts=contact_matrix(model,data,.002)
        self.assertEqual(len(contacts),4)
        result=fit_required_force(data.qfrc_bias,matrix,np.eye(6)[:,0:1],[-1e-10],[1e-10],np.ones(6))
        np.testing.assert_allclose(result['residual'],0.,atol=1e-6)
        data.qpos[2]=1.;mujoco.mj_forward(model,data)
        matrix,contacts=contact_matrix(model,data,.002)
        self.assertEqual(len(contacts),0)
        result=fit_required_force(data.qfrc_bias,matrix,np.eye(6)[:,0:1],[-1e-10],[1e-10],np.ones(6))
        self.assertAlmostEqual(result['residual'][2],-19.62)

    def test_friction_and_motor_limit_cannot_be_bypassed(self):
        # One normal-only contact cannot explain a horizontal wrench, and a
        # 2 N actuator cannot supply a required 3 N force.
        result=fit_required_force(np.array([3.,10.]),np.array([[0.],[1.]]),np.array([[1.],[0.]]),[-2.],[2.],np.ones(2))
        np.testing.assert_allclose(result['residual'],[-1.,0.],atol=1e-7)
        self.assertAlmostEqual(result['torques'][0],2.)
        fixed=fit_required_force(np.array([3.,10.]),np.array([[0.],[1.]]),np.array([[1.],[0.]]),[2.],[2.],np.ones(2))
        np.testing.assert_allclose(fixed['residual'],[-1.,0.],atol=1e-7)
        self.assertEqual(fixed['torques'][0],2.)
        unsupported=fit_required_force(np.array([3.,10.]),np.zeros((2,0)),np.eye(2),[2.,0.],[2.,0.],np.ones(2))
        np.testing.assert_array_equal(unsupported['residual'],[-1.,-10.])


if __name__=='__main__':unittest.main()
