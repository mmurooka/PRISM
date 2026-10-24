"""Collision-only counterfactuals for simulated contact evaluation."""
import numpy as np


def disable_box_weld_contact(model, box_geom, weld):
    """Remove one articulated body's box contact while preserving other pairs.

    Supports the local scene's binary collision masks. It verifies all pairwise
    masks before changing the model, excluding same-weld pairs which MuJoCo
    already suppresses. This is an artificial ablation, never a physical scene.
    """
    c=model.geom_contype.copy();a=model.geom_conaffinity.copy()
    if not np.all(((c==0)&(a==0))|((c==1)&(a==1))):
        raise ValueError('Counterfactual requires standard 0/0 or 1/1 masks')
    welds=model.body_weldid[model.geom_bodyid]
    if weld in (0,welds[box_geom]):raise ValueError('Select a robot articulated body')
    selected=(welds==weld)&(c!=0)
    if not selected.any():raise ValueError('No collidable geometry on selected body')
    def allowed(c,a):return ((c[:,None]&a[None,:])|(a[:,None]&c[None,:]))!=0
    before=allowed(c,a);c[box_geom]=2;a[box_geom]=1;c[selected]=0;a[selected]=1
    after=allowed(c,a);different_weld=welds[:,None]!=welds[None,:]
    expected=np.zeros_like(before);expected[box_geom,selected]=True;expected[selected,box_geom]=True
    removed=before&~after&different_weld;added=~before&after&different_weld
    if added.any() or not np.array_equal(removed,expected):
        raise ValueError('Intervention would change unrelated collision pairs')
    model.geom_contype[:]=c;model.geom_conaffinity[:]=a
    return dict(box_geom=int(box_geom),robot_weld=int(weld),robot_geoms=np.flatnonzero(selected).tolist(),
                removed_unordered_pairs=int(removed.sum()//2),added_pairs=0)
