import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.adapters import assemble_input,load_interactvlm_contacts


class AdapterTests(unittest.TestCase):
    def fixture(self):
        vertices=np.zeros((3,6890,3))
        weights=np.zeros((6890,24)); weights[:,0]=1
        joints=np.zeros((24,3)); joints[[1,2],2]=1; joints[[7,8],2]=-1
        human=dict(human_vertices_world=vertices,template_vertices=vertices[0],
                   skinning_weights=weights,rest_joints=joints,timestamps_s=[0,.03,.07],
                   world_frame='calibrated',source_video_sha256='test',length_unit='m',
                   scale_source='test',body_topology='SMPL_6890_from_SMPLX_mapping')
        objects=dict(timestamps_s=[0,.03,.07],world_frame='calibrated',source_video_sha256='test',
                     length_unit='m',object_id='box',box_half_extents=[.2,.3,.4],
                     world_from_object=np.tile(np.eye(4),(3,1,1)))
        return human,objects

    def test_join_and_original_contact_frame_ids(self):
        human,objects=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            for frame,value in [(10,0),(12,1),(15,.5)]:
                np.savez(Path(tmp)/f'{frame:05d}.npz',pred_contact_3d_smplh=np.full(6890,value))
            scores=load_interactvlm_contacts(tmp,[10,12,15])
            self.assertEqual(scores[:,0].tolist(),[0,1,.5])
            with self.assertRaises(FileNotFoundError): load_interactvlm_contacts(tmp,[10,11,12])
        result=assemble_input(human,objects,scores,[0,-9.81,0])
        self.assertEqual(result['world_frame'],'calibrated')
        np.testing.assert_array_equal(result['gravity_world'],[0,-9.81,0])

    def test_coordinate_provenance_timestamp_and_missing_pose_rejected(self):
        for field,value in [('world_frame','uncalibrated'),('source_video_sha256','another_video'),
                            ('length_unit','cm'),('timestamps_s',[0,.04,.08]),('valid',[True,False,True])]:
            h,o=self.fixture(); o[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                assemble_input(h,o,np.zeros((3,6890)),[0,-9.81,0])


if __name__=='__main__': unittest.main()
