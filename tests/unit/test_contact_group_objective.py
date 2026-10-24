import unittest
import torch

from prism_reconstruction.contact_group_objective import alternative_patch_loss


class AlternativePatchTest(unittest.TestCase):
    def test_alternatives_and_missing_frames(self):
        distances=torch.tensor([.01**2,.20**2,.02**2,10.],requires_grad=True)
        frames=torch.tensor([0,0,3,4]);selected=torch.tensor([True,True,True,False])
        loss=alternative_patch_loss(distances,frames,selected,scale=.01)
        self.assertAlmostEqual(float(loss),2.5)
        loss.backward()
        self.assertEqual(float(distances.grad[1]),0.)
        self.assertEqual(float(distances.grad[3]),0.)
        self.assertGreater(float(distances.grad[0]),0.)

    def test_empty_selection_keeps_zero_gradient(self):
        distances=torch.tensor([1.],requires_grad=True)
        loss=alternative_patch_loss(distances,torch.tensor([0]),torch.tensor([False]))
        loss.backward()
        self.assertEqual(float(loss),0.)
        self.assertEqual(float(distances.grad[0]),0.)


if __name__=='__main__':unittest.main()
