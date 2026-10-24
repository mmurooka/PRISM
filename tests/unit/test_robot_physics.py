import sys
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.robot_physics import prepare_physics_scene,motor_parameters


class RobotPhysicsTests(unittest.TestCase):
    def test_only_hinge_motors_actuate_and_free_object_obeys_gravity(self):
        names=[side+'_'+name+'_joint' for side in ('left','right') for name in
               ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        names += ['waist_'+name+'_joint' for name in ('yaw','roll','pitch')]
        names += [side+'_'+name+'_joint' for side in ('left','right') for name in
                  ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
        root=ET.Element('mujoco');world=ET.SubElement(root,'worldbody')
        robot=ET.SubElement(world,'body',name='pelvis',pos='-10 0 5')
        ET.SubElement(robot,'freejoint');ET.SubElement(robot,'geom',type='sphere',size='.02',mass='1')
        for i,name in enumerate(names):
            robot=ET.SubElement(robot,'body',name=f'link_{i}',pos='0 0 .05')
            ET.SubElement(robot,'joint',name=name,type='hinge',range='-2 2',actuatorfrcrange='-10 10')
            ET.SubElement(robot,'geom',type='sphere',size='.02',mass='1')
        box=ET.SubElement(world,'body',name='hoi_box',pos='0 0 2')
        ET.SubElement(box,'freejoint',name='hoi_box_free')
        ET.SubElement(box,'geom',name='hoi_box_geom',type='box',size='.2 .2 .2',mass='1')
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'ik.xml';output=Path(temp)/'physics.xml';ET.ElementTree(root).write(source)
            prepare_physics_scene(source,output);model=mujoco.MjModel.from_xml_path(str(output));data=mujoco.MjData(model)
            motor=motor_parameters(model)
            self.assertEqual(model.nu,29);self.assertEqual(model.neq,0)
            self.assertTrue(np.all(model.jnt_type[model.actuator_trnid[:,0]]==mujoco.mjtJoint.mjJNT_HINGE))
            np.testing.assert_array_equal(motor['upper'],10.)
            self.assertTrue(np.all(motor['kp']>0))
            joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'hoi_box_free');index=model.jnt_qposadr[joint]
            for _ in range(50):mujoco.mj_step(model,data)
            np.testing.assert_allclose(data.qpos[index:index+2],0.,atol=1e-12)
            self.assertAlmostEqual(data.qpos[index+2],2-.5*9.81*data.time**2,delta=.002)
            self.assertTrue(np.all(data.qfrc_applied==0));self.assertTrue(np.all(data.xfrc_applied==0))


if __name__=='__main__':unittest.main()
