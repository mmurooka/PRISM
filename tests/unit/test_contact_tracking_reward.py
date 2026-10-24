import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.contact_tracking_env import contact_reward_terms


class ContactTrackingRewardTests(unittest.TestCase):
    def test_force_elsewhere_does_not_replace_target_location(self):
        near=contact_reward_terms([.01],[100.],[1.],'coupled')
        far=contact_reward_terms([.4],[100.],[1.],'coupled')
        absent=contact_reward_terms([.01],[0.],[1.],'coupled')
        self.assertGreater(near[1],.8);self.assertLess(far[1],.01)
        self.assertEqual(absent[1],0.);self.assertEqual(near[0],absent[0])
        self.assertEqual(near[2],far[2])  # Force-only diagnostic exposes its limitation.

    def test_dense_approach_and_legacy_formula(self):
        terms=[contact_reward_terms([distance],[.5],[1.],'coupled') for distance in [.03,.08,.15]]
        self.assertTrue(terms[0][0]>terms[1][0]>terms[2][0]>0)
        for position,gated,_ in terms:self.assertAlmostEqual(gated,.5*position)
        position,force,_=contact_reward_terms([.03],[2.],[1.],'additive')
        self.assertAlmostEqual(position,np.exp(-1));self.assertEqual(force,1.)


if __name__=='__main__':unittest.main()
