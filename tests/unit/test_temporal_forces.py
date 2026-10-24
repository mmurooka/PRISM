import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.dynamics import ForceCandidate
from prism_reconstruction.temporal_forces import solve_force_sequence


class TemporalForceTests(unittest.TestCase):
    def test_smoothing_reduces_noisy_support_forces_and_reports_residual(self):
        t=np.arange(30)/30
        wrench=np.zeros((30,6)); wrench[:,2]=9.81+.25*(-1.)**np.arange(30)
        candidates=[[ForceCandidate('left_knee',np.zeros(3),np.array([0,0,1]),friction=.3)] for _ in t]
        raw=solve_force_sequence(candidates,wrench,np.zeros((30,3)),t,1,smoothness=0)
        smooth=solve_force_sequence(candidates,wrench,np.zeros((30,3)),t,1,smoothness=.005)
        f0=raw['region_forces_world'][:,0]
        f1=smooth['region_forces_world'][:,0]
        self.assertLess(np.linalg.norm(np.diff(f1,axis=0)),.5*np.linalg.norm(np.diff(f0,axis=0)))
        self.assertTrue(smooth['feasible'].all())
        self.assertGreater(smooth['normalized_residual'].mean(),raw['normalized_residual'].mean())
        self.assertTrue((np.linalg.norm(f1[:,:2],axis=1)<=.3*f1[:,2]+1e-8).all())

    def test_missing_contact_is_not_filled_by_temporal_prior(self):
        t=np.arange(5)/30
        candidate=ForceCandidate('hand',np.zeros(3),np.array([0,0,1]))
        result=solve_force_sequence([[candidate],[candidate],[],[candidate],[candidate]],
            np.tile([0,0,9.81,0,0,0],(5,1)),np.zeros((5,3)),t,1)
        self.assertEqual(result['forces_world'][2].shape,(0,3))
        np.testing.assert_allclose(result['region_forces_world'][2],0)
        self.assertFalse(result['feasible'][2])


if __name__=='__main__': unittest.main()
