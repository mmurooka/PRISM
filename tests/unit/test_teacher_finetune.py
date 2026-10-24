import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prism_reconstruction.teacher_finetune import initialize_actor_from_teacher


class TeacherFinetuneTests(unittest.TestCase):
    def test_new_contact_inputs_preserve_actor_and_remain_trainable(self):
        torch.manual_seed(4)
        teacher=torch.nn.Sequential(torch.nn.Linear(178,32),torch.nn.ELU(),torch.nn.Linear(32,16),torch.nn.ELU(),torch.nn.Linear(16,29))
        trunk=torch.nn.Sequential(torch.nn.Linear(250,32),torch.nn.ELU(),torch.nn.Linear(32,16),torch.nn.ELU())
        head=torch.nn.Linear(16,29)
        policy=SimpleNamespace(mlp_extractor=SimpleNamespace(policy_net=trunk),action_net=head)
        initialize_actor_from_teacher(policy,teacher)
        samples=torch.randn(100,250);before=teacher(samples[:,:178]);after=head(trunk(samples))
        torch.testing.assert_close(before,after,atol=2e-7,rtol=1e-6)
        after.square().sum().backward()
        self.assertGreater(trunk[0].weight.grad[:,178:].abs().sum().item(),0.)


if __name__=='__main__':unittest.main()
