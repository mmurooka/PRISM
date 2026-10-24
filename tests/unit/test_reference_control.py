import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from prism_reconstruction.reference_control import load_inverse_torques


class TorqueProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.scene=self.root/'scene.xml'
        (self.root/'trajectory.npz').write_bytes(b'trajectory');self.scene.write_bytes(b'physics')
        self.path=self.root/'gap_0.0020.npz'
        np.savez(self.path,torques=np.zeros((2,3)))
        self.path.with_suffix('.json').write_text(json.dumps([{'frame':0},{'frame':1}]))
        (self.root/'metadata.json').write_text(json.dumps(dict(
            source_trajectory_sha256=hashlib.sha256(b'trajectory').hexdigest(),
            physics_xml_sha256=hashlib.sha256(b'physics').hexdigest())))

    def load(self):return load_inverse_torques(self.path,self.root,self.scene,2,3)

    def test_complete_matching_audit(self):
        self.assertEqual(self.load().shape,(2,3))

    def test_changed_physics_rejected(self):
        self.scene.write_bytes(b'different mass')
        with self.assertRaisesRegex(ValueError,'physics_xml_sha256'):self.load()

    def test_sparse_frames_rejected(self):
        self.path.with_suffix('.json').write_text(json.dumps([{'frame':0},{'frame':2}]))
        with self.assertRaisesRegex(ValueError,'complete per-frame'):self.load()


if __name__=='__main__':unittest.main()
