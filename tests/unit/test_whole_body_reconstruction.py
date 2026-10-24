"""Analytic tests, runnable without Isaac Sim or trained checkpoints."""
import sys
from pathlib import Path
import unittest
import tempfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial.transform import Rotation
from prism_reconstruction.geometry import box_surface, smpl_vertex_regions
from prism_reconstruction.contacts import ContactConfig, infer_contacts
from prism_reconstruction.dynamics import ForceCandidate, floor_candidates, required_wrench, solve_contact_forces
from prism_reconstruction.tracking import rigid_transform, track_rigid_object, estimate_box_geometry
from prism_reconstruction.pipeline import run_archive, reconstruct_contacts


class WholeBodyContactTests(unittest.TestCase):
    def test_surface_inside_outside_and_normal(self):
        closest, normals, gap = box_surface([[0, 0, -0.51], [0.49, 0, 0], [0, 0.6, 0]], [.5]*3)
        np.testing.assert_allclose(gap, [.01, -.01, .1])
        np.testing.assert_allclose(closest, [[0, 0, -.5], [.5, 0, 0], [0, .5, 0]])
        np.testing.assert_allclose(normals, [[0, 0, -1], [1, 0, 0], [0, 1, 0]])

    def test_two_hands_and_knee_follow_moving_object(self):
        times = np.arange(12)/30
        poses = np.tile(np.eye(4), (len(times), 1, 1))
        poses[:, :3, :3] = Rotation.from_euler('z', times).as_matrix()
        poses[:, 0, 3] = times
        local = np.array([[-.3, -.2, -.5], [.3, -.2, -.5], [0, .3, -.5]])
        human = np.einsum('tij,vj->tvi', poses[:, :3, :3], local) + poses[:, None, :3, 3]
        result = infer_contacts(human, poses, [.5]*3, np.ones((12, 3)),
                                ['left_hand', 'right_hand', 'left_knee'], times,
                                ContactConfig(min_region_vertices=1))
        self.assertTrue(result['active'].all())
        np.testing.assert_allclose(result['object_points'], np.broadcast_to(local, (12, 3, 3)), atol=1e-6)
        self.assertEqual({i['region'] for i in result['intervals']}, {'left_hand', 'right_hand', 'left_knee'})
        self.assertTrue((result['mode'] == 1).all())

    def test_visual_false_positive_and_one_frame_flicker_rejected(self):
        times = np.arange(6)/30
        poses = np.tile(np.eye(4), (6, 1, 1))
        human = np.tile([[0, 0, -.5], [3, 0, 0]], (6, 1, 1))
        scores = np.ones((6, 2)); scores[:, 0] = 0; scores[2, 0] = 1
        result = infer_contacts(human, poses, [.5]*3, scores, ['left_knee', 'left_hand'], times)
        self.assertFalse(result['active'].any())

    def test_knee_is_a_distinct_region(self):
        joints = np.zeros((24, 3))
        joints[[1,4,7]] = [[.2,0,0], [.2,0,-.4], [.2,0,-.8]]
        joints[[2,5,8]] = [[-.2,0,0], [-.2,0,-.4], [-.2,0,-.8]]
        vertices = np.array([[.2,.04,-.4], [-.2,.04,-.4], [.2,0,-.7]])
        weights = np.zeros((3, 24)); weights[0,4] = weights[1,5] = weights[2,4] = 1
        labels = smpl_vertex_regions(vertices, weights, joints)
        self.assertEqual(labels.tolist(), ['left_knee', 'right_knee', 'left_shin'])

    def test_frame_mismatch_and_invalid_probability_fail(self):
        poses = np.tile(np.eye(4), (3,1,1))
        with self.assertRaises(ValueError):
            infer_contacts(np.zeros((3,2,3)), poses, [.5]*3, np.ones((2,2)), ['a','b'], [0, .1, .2])
        with self.assertRaises(ValueError):
            infer_contacts(np.zeros((3,2,3)), poses, [.5]*3, np.full((3,2), 2), ['a','b'], [0, .1, .2])


class DynamicsTests(unittest.TestCase):
    def test_gravity_and_freefall(self):
        times = np.arange(20)/30
        poses = np.tile(np.eye(4), (20,1,1))
        gravity = np.array([0,0,-9.81])
        static = required_wrench(poses, times, [.5]*3, gravity)
        np.testing.assert_allclose(static['wrench_world'], np.tile([0,0,9.81,0,0,0], (20,1)), atol=1e-9)
        poses[:, :3, 3] = .5*times[:,None]**2*gravity
        falling = required_wrench(poses, times, [.5]*3, gravity)
        np.testing.assert_allclose(falling['wrench_world'], 0, atol=1e-9)

    def test_rotational_acceleration(self):
        times = np.arange(20)/30
        poses = np.tile(np.eye(4), (20,1,1))
        poses[:, :3, :3] = Rotation.from_euler('z', .5*2*times**2).as_matrix()
        result = required_wrench(poses, times, [.5]*3, [0,0,-9.81])
        np.testing.assert_allclose(result['alpha_world'][:,2], 2, atol=1e-8)
        np.testing.assert_allclose(result['wrench_world'][:,5], 1/3, atol=1e-8)

    def test_two_hands_and_knee_are_needed_for_balance(self):
        candidates = [ForceCandidate(region, np.array(point), np.array([0,0,1]), friction=0)
                      for region, point in [('left_hand', [-.3,-.2,-.5]),
                                            ('right_hand', [.3,-.2,-.5]),
                                            ('left_knee', [0,.3,-.5])]]
        result = solve_contact_forces(candidates, [0,0,9.81,0,0,0], [0,0,0], 1)
        self.assertTrue(result['feasible'])
        self.assertTrue(result['necessary'].all())
        self.assertTrue((result['forces_world'][:,2] > 1).all())
        self.assertGreater(result['residual_without_region']['left_knee'], .08)

    def test_floor_support_and_unexplained_force(self):
        pose = np.eye(4); pose[2,3] = .5
        candidates = floor_candidates(pose, [.5]*3, [0,0,1], 0)
        result = solve_contact_forces(candidates, [0,0,9.81,0,0,0], [0,0,.5], 1)
        self.assertTrue(result['feasible'])
        self.assertTrue(all(c.source == 'scene' for c in candidates))
        unsupported = solve_contact_forces([], [0,0,9.81,0,0,0], [0,0,0], 1)
        self.assertFalse(unsupported['feasible'])
        self.assertEqual(unsupported['forces_world'].shape, (0,3))

    def test_ambiguity_is_not_reported_as_necessary_contact(self):
        candidates = [ForceCandidate(name, np.zeros(3), np.array([0,0,1]), friction=0)
                      for name in ['left_hand', 'left_knee']]
        result = solve_contact_forces(candidates, [0,0,9.81,0,0,0], [0,0,0], 1)
        self.assertTrue(result['feasible'])
        self.assertFalse(result['necessary'].any())

    def test_friction_limit(self):
        candidate = ForceCandidate('hand', np.zeros(3), np.array([0,0,1]), friction=.2)
        result = solve_contact_forces([candidate], [5,0,10,0,0,0], [0,0,0], 1)
        self.assertFalse(result['feasible'])
        force = result['forces_world'][0]
        self.assertLessEqual(np.linalg.norm(force[:2]), .2*force[2]+1e-8)

    def test_knee_thigh_ambiguity_retains_group_necessity(self):
        candidates=[ForceCandidate(region,np.array(point),np.array([0,0,1]),friction=0)
                    for region,point in [('left_hand',[-.3,-.2,-.5]),('right_hand',[.3,-.2,-.5]),
                                         ('right_knee',[0,.3,-.5]),('right_thigh',[0,.3,-.5])]]
        result=solve_contact_forces(candidates,[0,0,9.81,0,0,0],[0,0,0],1,
                                   removal_groups={'right_knee_and_thigh':['right_knee','right_thigh']})
        self.assertTrue(result['feasible'])
        self.assertFalse(result['necessary'][2:].any())
        self.assertTrue(result['necessary_groups']['right_knee_and_thigh'])
        unsupported=solve_contact_forces(candidates,[0,0,-9.81,0,0,0],[0,0,0],1,
                                        removal_groups={'right_knee_and_thigh':['right_knee','right_thigh']})
        self.assertFalse(unsupported['feasible'])
        self.assertFalse(unsupported['necessary_groups']['right_knee_and_thigh'])
        empty=solve_contact_forces([],[0,0,9.81,0,0,0],[0,0,0],1,
                                   removal_groups={'right_knee_and_thigh':['right_knee','right_thigh']})
        self.assertFalse(empty['feasible'])
        self.assertFalse(empty['necessary_groups']['right_knee_and_thigh'])


class RigidTrackingTests(unittest.TestCase):
    def test_robust_tracks_and_occlusion_failure(self):
        rng = np.random.default_rng(5)
        reference = rng.uniform(-.5,.5,(40,3))
        poses = np.tile(np.eye(4), (3,1,1))
        poses[1,:3,:3] = Rotation.from_euler('zy', [.4,.2]).as_matrix()
        poses[1,:3,3] = [.3,.1,.2]
        points = np.einsum('tij,nj->tni', poses[:,:3,:3], reference) + poses[:,None,:3,3]
        points[1,:8] += 3
        visibility = np.ones((3,40)); visibility[2,:37] = 0
        result = track_rigid_object(points, visibility)
        self.assertEqual(result['valid'].tolist(), [True, True, False])
        np.testing.assert_allclose(result['world_from_reference'][1], poses[1], atol=1e-8)
        self.assertTrue(np.isnan(result['world_from_reference'][2]).all())

    def test_degenerate_tracks_fail(self):
        points = np.c_[np.arange(5), np.zeros((5,2))]
        with self.assertRaises(ValueError):
            rigid_transform(points, points)

    def test_box_dimensions_from_observed_surfaces(self):
        from itertools import product
        points = np.array(list(product([-1,0,1], repeat=3))) * [.5,.3,.2]
        result = estimate_box_geometry(np.eye(4)[None], points[None])
        np.testing.assert_allclose(np.sort(result['half_extents']), [.2,.3,.5], atol=1e-8)


class ReconstructionArchiveTests(unittest.TestCase):
    def test_dynamics_candidate_preserves_missing_visual_evidence(self):
        times=np.arange(6)/30
        vertices=np.tile([[-.3,-.2,-.5],[.3,-.2,-.5],[0,.3,-.5]],(6,1,1))
        data=dict(timestamps_s=times,human_vertices_world=vertices,world_from_object=np.tile(np.eye(4),(6,1,1)),
                  box_half_extents=[.5]*3,human_vertex_regions=['left_hand','right_hand','right_knee'],
                  contact_scores=np.tile([1.,1.,0.],(6,1)),gravity_world=[0,0,-9.81],length_unit='m',
                  world_frame='synthetic_world',scale_source='synthetic_known',source_video_sha256='0'*64,object_id='box')
        arrays,metadata,records=reconstruct_contacts(data,config=ContactConfig(min_region_vertices=1))
        self.assertFalse(arrays['active'][:,2].any())
        self.assertTrue(arrays['contact_candidates'][:,2].all())
        self.assertTrue(((arrays['contact_evidence_code'][:,2]&1)==0).all())
        self.assertTrue(((arrays['contact_evidence_code'][:,2]&4)>0).all())
        self.assertIn('not independent contact labels',metadata['candidate_interpretation'])

    def test_complete_archive_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            times = np.arange(6)/30
            human = np.tile([[-.3,-.2,-.5], [.3,-.2,-.5], [0,.3,-.5]], (6,1,1))
            source = directory/'input.npz'
            np.savez(source, timestamps_s=times, human_vertices_world=human,
                     world_from_object=np.tile(np.eye(4),(6,1,1)), box_half_extents=[.5]*3,
                     human_vertex_regions=['left_hand','right_hand','left_knee'],
                     contact_scores=np.ones((6,3)), gravity_world=[0,0,-9.81],
                     length_unit='m', world_frame='synthetic_world', scale_source='synthetic_known',
                     source_video_sha256='0'*64, object_id='box')
            output = directory/'output'
            metadata = run_archive(source, output, config=ContactConfig(min_region_vertices=1))
            self.assertEqual(metadata['force_unit'], 'N_per_kg')
            self.assertEqual(metadata['scene_support'], 'not_provided')
            self.assertEqual(len(metadata['input_sha256']), 64)
            with np.load(output/'contacts.npz', allow_pickle=False) as archive:
                self.assertTrue(archive['active'].all())
                self.assertIn('left_knee', archive['region_names'])
                self.assertTrue(np.all(archive['normalized_wrench_residual'] < .08))
            evidence = json.loads((output/'force_evidence.json').read_text())
            self.assertEqual(len(evidence), 6)
            self.assertGreater(evidence[0]['hand_only_residual'], evidence[0]['normalized_residual'])
            with self.assertRaises(FileExistsError):
                run_archive(source, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
