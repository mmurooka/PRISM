import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.robot_model_profile import aggregate_urdf_inertia

class RobotProfileTests(unittest.TestCase):
    def test_fixed_child_mass_and_parallel_axis_are_counted_once(self):
        xml='''<robot name="test"><link name="base"><inertial><mass value="2"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial></link>
        <link name="payload"><inertial><mass value="1"/><inertia ixx=".2" iyy=".3" izz=".4" ixy="0" ixz="0" iyz="0"/></inertial></link>
        <joint name="fixed" type="fixed"><parent link="base"/><child link="payload"/><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'robot.urdf';p.write_text(xml);groups,limits=aggregate_urdf_inertia(p)
        self.assertEqual(list(groups),['base']);self.assertEqual(limits,{})
        self.assertEqual(groups['base']['mass'],3)
        np.testing.assert_allclose(groups['base']['com'],[1/3,0,0])
        np.testing.assert_allclose(groups['base']['inertia'],np.diag([.4,.3+2/3,.5+2/3]),atol=1e-14)

if __name__=='__main__':unittest.main()
