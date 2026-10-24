import copy
import unittest
import numpy as np

from prism_reconstruction.semantic_prior import augment_binary_prior


class SemanticPriorTests(unittest.TestCase):
    def setUp(self):
        self.data=dict(contact_prior_source='interactvlm', source_video_sha256='video',
            timestamps_s=np.arange(3)/30, human_vertex_regions=np.array(['left_hand','right_thigh']),
            contact_scores=np.array([[1,0],[1,0],[0,0]],np.float32))
        self.semantic=dict(timestamps_s=self.data['timestamps_s'].copy(),
            human_vertex_regions=self.data['human_vertex_regions'].copy(),
            fused_vertex_candidate=np.array([[0,0],[0,1],[0,1]],bool))
        self.meta=dict(complete=True,source_video_sha256='video',source_contact_input_sha256='base')

    def test_union_preserves_old_masks_and_separate_sources(self):
        original=self.data['contact_scores'].copy()
        result=augment_binary_prior(self.data,self.semantic,self.meta,'base')
        np.testing.assert_array_equal(result['contact_prior_evidence_code'],[[1,0],[1,2],[0,2]])
        np.testing.assert_array_equal(result['contact_scores'],[[1,0],[1,1],[0,1]])
        np.testing.assert_array_equal(result['contact_scores_interactvlm'],original)
        np.testing.assert_array_equal(self.data['contact_scores'],original)

    def test_wrong_geometry_input_or_vertex_order_is_rejected(self):
        with self.assertRaises(ValueError):augment_binary_prior(self.data,self.semantic,self.meta,'different')
        altered=copy.deepcopy(self.semantic);altered['human_vertex_regions']=altered['human_vertex_regions'][::-1]
        with self.assertRaises(ValueError):augment_binary_prior(self.data,altered,self.meta,'base')

    def test_continuous_scores_are_not_silently_binarized(self):
        self.data['contact_scores'][0,0]=.7
        with self.assertRaises(ValueError):augment_binary_prior(self.data,self.semantic,self.meta,'base')

    def test_source_identity_survives_dynamics_and_retarget_selection(self):
        from prism_reconstruction.contacts import ContactConfig
        from prism_reconstruction.pipeline import reconstruct_contacts
        from prism_reconstruction.retargeting import select_contact_targets
        data=dict(timestamps_s=np.arange(6)/30,
            human_vertices_world=np.tile([[-.3,-.2,-.5],[.3,-.2,-.5],[0,.3,-.5]],(6,1,1)),
            world_from_object=np.tile(np.eye(4),(6,1,1)),box_half_extents=[.5]*3,
            human_vertex_regions=np.array(['left_hand','right_hand','right_knee']),
            contact_scores=np.ones((6,3)),gravity_world=[0,0,-9.81],length_unit='m',
            world_frame='synthetic',scale_source='known',source_video_sha256='0'*64,object_id='box',
            contact_prior_source='interactvlm_plus_local_language_image_geometry',
            contact_prior_evidence_code=np.tile([1,1,2],(6,1)).astype(np.uint8),
            contact_prior_base_input_sha256='base',contact_prior_semantic_cues_sha256='cues',
            contact_prior_semantic_metadata_sha256='metadata',contact_prior_score_interpretation='binary candidates')
        arrays,meta,records=reconstruct_contacts(data,config=ContactConfig(min_region_vertices=1))
        np.testing.assert_array_equal(arrays['contact_prior_evidence_code'],data['contact_prior_evidence_code'])
        targets=select_contact_targets(records,arrays,meta)
        knee=next(t for t in targets[0] if t['region']=='right_knee')
        self.assertEqual(knee['visual_prior_sources'],['local_language_image_geometry'])
        self.assertNotIn('interactvlm',knee['visual_prior_sources'])
        self.assertEqual(meta['evidence_code_bits']['1'],'combined_binary_prior_and_geometric_contact')


if __name__=='__main__':unittest.main()
