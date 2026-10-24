import unittest
import numpy as np
from prism_reconstruction.contact_patch import contact_patch_forces


class PhysicalContactPatchTest(unittest.TestCase):
    def test_remote_same_body_force_is_not_matched(self):
        contacts=[dict(other_weld=7,point_object=[.3,0,0],normal_force_n=20.)]
        smooth,matched,distance=contact_patch_forces(contacts,7,np.zeros(3))
        self.assertEqual(matched,0.)
        self.assertLess(smooth,1e-30)
        self.assertAlmostEqual(distance,.3)

    def test_nearby_forces_sum_without_other_bodies(self):
        contacts=[dict(other_weld=7,point_object=[.01,0,0],normal_force_n=.6),
                  dict(other_weld=7,point_object=[-.01,0,0],normal_force_n=.6),
                  dict(other_weld=8,point_object=[0,0,0],normal_force_n=100.)]
        smooth,matched,distance=contact_patch_forces(contacts,7,np.zeros(3))
        self.assertAlmostEqual(matched,1.2)
        self.assertLess(smooth,matched)
        self.assertGreater(smooth,1.)
        self.assertAlmostEqual(distance,.01)


if __name__=='__main__':unittest.main()
