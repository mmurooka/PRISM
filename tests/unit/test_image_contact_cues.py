import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.image_contact_cues import boundary_contact_cues,persistent_region_cues

class ImageCueTests(unittest.TestCase):
    def test_image_adjacency_does_not_override_depth_gap(self):
        person=np.zeros((40,40),np.uint8);obj=person.copy();person[10:30,10:20]=1;obj[10:30,20:30]=1
        vertices=np.array([[19.,20.,1.],[19.,20.,1.],[-5.,20.,1.],[19.,20.,-1.]])
        result=boundary_contact_cues(vertices,np.eye(3),person,obj,np.array([0.,.2,0.,0.]))
        np.testing.assert_array_equal(result['corroborated_candidate'],[True,False,False,False])
        np.testing.assert_array_equal(result['image_supported'],[True,True,False,False])
        none=boundary_contact_cues(vertices,np.eye(3),person,np.zeros_like(obj),np.zeros(4))
        self.assertFalse(none['corroborated_candidate'].any());self.assertTrue(np.isfinite(none['boundary_distance_px']).all())

    def test_region_cues_require_persistence(self):
        a=np.zeros((8,4),bool);a[0:2,:3]=1;a[4:7,:3]=1
        names,counts,active=persistent_region_cues(a,np.array(['knee']*4))
        np.testing.assert_array_equal(active[:,0],[False,False,False,False,True,True,True,False])

if __name__=='__main__':unittest.main()
