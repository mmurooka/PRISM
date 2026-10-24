"""Match actual simulated contact forces to an object-local target patch."""
import numpy as np


def contact_patch_forces(contacts,weld,target,sigma=.03,radius=.03):
    """Return smooth force guidance, hard matched force, nearest loaded point.

    All locations are in object coordinates. A force elsewhere on the same
    articulated body cannot satisfy the hard target-location criterion.
    """
    if sigma<=0 or radius<=0:raise ValueError('Positive patch scales required')
    selected=[c for c in contacts if c['other_weld']==weld and c['normal_force_n']>.01]
    if not selected:return 0.,0.,None
    distance=np.array([np.linalg.norm(np.asarray(c['point_object'])-target) for c in selected])
    force=np.array([c['normal_force_n'] for c in selected])
    return (float(np.sum(force*np.exp(-(distance/sigma)**2))),
            float(force[distance<radius].sum()),float(distance.min()))
