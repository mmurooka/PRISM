"""Preserve the released actor while adding initially neutral contact inputs."""
import torch


@torch.no_grad()
def initialize_actor_from_teacher(policy,teacher_actor):
    """Copy 178-input ELU actor into SB3 actor with extra zero-weight features."""
    source=[layer for layer in teacher_actor if isinstance(layer,torch.nn.Linear)]
    target=[layer for layer in policy.mlp_extractor.policy_net if isinstance(layer,torch.nn.Linear)]
    target.append(policy.action_net)
    if len(source)!=len(target) or source[0].in_features!=178:
        raise ValueError('Unexpected teacher/SB3 layer structure')
    if any(not isinstance(layer,(torch.nn.Linear,torch.nn.ELU)) for layer in policy.mlp_extractor.policy_net):
        raise ValueError('ELU policy layers required for teacher parity')
    for index,(old,new) in enumerate(zip(source,target)):
        if old.out_features!=new.out_features or (index and old.in_features!=new.in_features):
            raise ValueError('Teacher/SB3 hidden dimensions do not match')
        if index==0:
            if new.in_features<178:raise ValueError('Teacher observations must be the first 178 features')
            new.weight.zero_();new.weight[:,:178].copy_(old.weight.to(new.weight.device))
        else:new.weight.copy_(old.weight.to(new.weight.device))
        new.bias.copy_(old.bias.to(new.bias.device))
