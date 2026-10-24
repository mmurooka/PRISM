import unittest
import mujoco
import numpy as np
from prism_reconstruction.contact_intervention import disable_box_weld_contact


class ContactInterventionTest(unittest.TestCase):
    def model(self):
        return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="box" pos="0 0 1"><freejoint/><geom name="box_g" type="box" size=".1 .1 .1"/></body>
        <body name="leg" pos="1 0 1"><freejoint/><geom name="leg_g" size=".1"/>
          <body pos="0 0 .2"><geom name="fixed_g" size=".1"/></body></body>
        <body name="other" pos="2 0 1"><freejoint/><geom name="other_g" size=".1"/></body>
        </worldbody></mujoco>''')

    def test_only_box_leg_pairs_removed(self):
        m=self.model();box=m.geom('box_g').id;weld=m.body_weldid[m.body('leg').id]
        result=disable_box_weld_contact(m,box,weld)
        self.assertEqual(result['removed_unordered_pairs'],2)
        def collides(a,b):return bool((m.geom_contype[a]&m.geom_conaffinity[b])|(m.geom_contype[b]&m.geom_conaffinity[a]))
        self.assertFalse(collides(box,m.geom('leg_g').id))
        self.assertTrue(collides(m.geom('floor').id,m.geom('leg_g').id))
        self.assertTrue(collides(box,m.geom('other_g').id))

    def test_unknown_mask_rejected_without_mutation(self):
        m=self.model();m.geom_contype[0]=4;before=m.geom_contype.copy()
        with self.assertRaises(ValueError):disable_box_weld_contact(m,m.geom('box_g').id,m.body('leg').id)
        np.testing.assert_array_equal(m.geom_contype,before)


if __name__=='__main__':unittest.main()
