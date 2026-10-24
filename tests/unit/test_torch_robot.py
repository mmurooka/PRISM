import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.torch_robot import TorchRobot


class TorchRobotTests(unittest.TestCase):
    def test_fk_and_gradient_match_mujoco_with_offset_hinge_and_rotated_link(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="pelvis"><freejoint/><geom type="sphere" size=".1"/>
            <body pos=".2 .1 -.3" quat=".9238795 0 .3826834 0">
              <joint type="hinge" axis="1 0 0" pos=".02 .03 -.04"/>
              <geom type="box" size=".1 .2 .3"/>
              <body pos="0 0 -.2"><geom type="sphere" size=".03"/></body>
            </body>
          </body><body name="hoi_box"><freejoint/><geom type="box" size=".2 .3 .4"/></body>
          </worldbody></mujoco>''')
        data=mujoco.MjData(model);robot=TorchRobot(model)
        q=model.qpos0.copy();q[:3]=[.3,-.2,1.]
        q[3:7]=Rotation.from_euler('xyz',[.1,.2,.3]).as_quat()[[3,0,1,2]];q[7]=.35;q[8:11]=[1.,0.,.5]
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        joint=torch.tensor([[q[7]]],dtype=torch.float64,requires_grad=True)
        p,R=robot.forward(joint,robot.tensor(q[:3][None]),robot.tensor(Rotation.from_quat(q[[4,5,6,3]]).as_matrix()[None]),
            robot.tensor(q[8:11][None]),torch.eye(3,dtype=torch.float64)[None])
        np.testing.assert_allclose(p.detach().numpy()[0],data.xpos,atol=1e-12)
        np.testing.assert_allclose(R.detach().numpy()[0],data.xmat.reshape(-1,3,3),atol=1e-12)
        for axis in range(3):
            gradient=torch.autograd.grad(p[0,3,axis],joint,retain_graph=True)[0].item()
            jac=np.zeros((3,model.nv));mujoco.mj_jacBody(model,data,jac,None,3)
            self.assertAlmostEqual(gradient,jac[axis,6],places=10)


if __name__=='__main__':unittest.main()
