import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.retargeting import transform_demonstration,select_contact_targets


class RetargetingConventionsTests(unittest.TestCase):
    def test_floor_height_scales_without_resizing_or_distorting_object(self):
        joints=np.zeros((2,22,3));joints[:,:,2]=1
        joints[:,1,1]=.1;joints[:,2,1]=-.1;joints[:,16,1]=.2;joints[:,17,1]=-.2
        objects=np.tile(np.eye(4),(2,1,1));objects[:,2,3]=[.5,1.]
        half=np.array([.2,.3,.5])
        objects[1,:3,:3]=Rotation.from_euler('xyz',[.3,.2,.1]).as_matrix()
        objects[1,2,3]=.5+np.sum(abs(objects[1,2,:3])*half)
        # Express the same scene in an arbitrary reconstruction camera frame.
        camera=Rotation.from_euler('xyz',[.4,.2,-.3]).as_matrix();shift=np.array([2.,-1.,.7])
        world=joints@camera.T+shift;objects[:,:3,:3]=camera@objects[:,:3,:3]
        objects[:,:3,3]=objects[:,:3,3]@camera.T+shift
        normal=camera[:,2];offset=normal@shift
        result=transform_demonstration(world,objects,half,normal,offset,.75)
        np.testing.assert_allclose(result['box_half_extents'],half)
        pose=result['nominal_object_poses'];height=pose[:,2,3]-np.sum(abs(pose[:,2,:3])*half,axis=1)
        np.testing.assert_allclose(height,[0,.375],atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(pose[:,:3,:3]),1,atol=1e-12)

    def test_group_assignment_is_soft_and_not_unique(self):
        candidate=dict(source='human',region='right_thigh',vertex_ids=[0,1,2],necessary=False,explanatory=True,
                       point_object=[0,0,-.5],normal_world=[0,0,1],force_world=[0,0,9.81])
        records=[dict(candidates=[candidate],necessary_groups={'right_knee_and_thigh':True})]
        data=dict(active=np.zeros((1,3),bool),world_from_object=np.eye(4)[None])
        meta=dict(support_groups={'right_knee_and_thigh':['right_knee','right_thigh']},contact_config={'min_region_vertices':3})
        result=select_contact_targets(records,data,meta)[0][0]
        self.assertEqual(result['region'],'right_thigh')
        self.assertTrue(result['evidence']['group_assignment'])
        self.assertFalse(result['assignment_unique'])
        self.assertEqual(result['weight'],.5)
        self.assertEqual(select_contact_targets(records,data,meta,'hands_only'),[[]])

    def test_nonunique_force_change_does_not_jump_object_face(self):
        base=dict(source='human',region='right_thigh',vertex_ids=[0,1,2],necessary=False,
            explanatory=True,normal_world=[0,0,1])
        a=dict(**base,point_object=[0,0,-.5],force_world=[0,0,2.])
        b=dict(**base,point_object=[.2,0,-.3],force_world=[0,0,1.])
        records=[dict(candidates=[a,b]),dict(candidates=[dict(a),dict(b,force_world=[0,0,20.])])]
        data=dict(active=np.zeros((2,3),bool),world_from_object=np.tile(np.eye(4),(2,1,1)))
        meta=dict(support_groups={},contact_config={'min_region_vertices':3})
        result=select_contact_targets(records,data,meta,'all_explanatory')
        self.assertEqual(result[0][0]['point_object'],result[1][0]['point_object'])
        self.assertFalse(result[1][0]['assignment_unique'])


if __name__=='__main__':unittest.main()
