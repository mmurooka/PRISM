"""Optional task priorities, separate from estimated contact confidence."""
import torch


def alternative_patch_loss(squared_distances, frame_indices, selected, scale=.005):
    """Mean closest-patch squared distance per active frame.

    Adjacent semantic patches are alternatives, not simultaneous requirements.
    Frames without an existing selected target contribute neither loss nor a new
    contact label. This objective must be reported as a supplied task prior.
    """
    if scale <= 0:
        raise ValueError('Positive contact distance scale required')
    if not selected.any():
        return squared_distances.sum()*0
    _, inverse = torch.unique(frame_indices[selected], return_inverse=True)
    values = squared_distances[selected]
    closest = torch.full((int(inverse.max())+1,), float('inf'),
                         dtype=values.dtype, device=values.device)
    closest = closest.scatter_reduce(0, inverse, values, reduce='amin', include_self=True)
    return closest.mean()/scale**2
