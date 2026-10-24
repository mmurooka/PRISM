"""Contact-preserving retarget input conventions, separate from robot IK/RL."""
from __future__ import annotations

import numpy as np
from .geometry import validate_poses


def floor_aligned_frame(joints_world,normal_world,offset_m):
    """Robot axes: initial human forward, left, and observed floor normal."""
    joints=np.asarray(joints_world,float);up=np.asarray(normal_world,float)
    if (joints.shape[1:]!=(22,3) or len(joints)==0 or not np.isfinite(joints).all() or
            up.shape!=(3,) or not np.isclose(np.linalg.norm(up),1) or not np.isfinite(offset_m)):
        raise ValueError('Finite SMPL-X body joints and a unit floor normal are required')
    left=(joints[0,1]-joints[0,2])+(joints[0,16]-joints[0,17])
    left-=up*(left@up)
    if np.linalg.norm(left)<1e-5:raise ValueError('Initial body heading is not observable')
    left/=np.linalg.norm(left);forward=np.cross(left,up)
    rotation=np.array([forward,left,up])
    pelvis=joints[0,0];origin=pelvis-up*(pelvis@up-offset_m)
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=-rotation@origin
    return transform


def transform_demonstration(joints_world,object_poses,half_extents,normal_world,offset_m,body_scale):
    """Scale human motion while retaining the physical box dimensions.

    The nominal object's lowest height is scaled, rather than its dimensions.
    Thus a resting box remains on the floor, and knee-height support transfers
    without shrinking the actual object. Contact IK must adapt the upper body.
    """
    if not np.isfinite(body_scale) or body_scale<=0:raise ValueError('Positive body scale required')
    object_poses=validate_poses(object_poses);half_extents=np.asarray(half_extents,float)
    if len(object_poses)!=len(joints_world) or half_extents.shape!=(3,) or not np.isfinite(half_extents).all() or np.any(half_extents<=0):
        raise ValueError('Aligned object poses and positive cuboid dimensions required')
    transform=floor_aligned_frame(joints_world,normal_world,offset_m)
    rotation=transform[:3,:3];translation=transform[:3,3]
    joints=(np.asarray(joints_world)@rotation.T+translation)*body_scale
    objects=transform@np.asarray(object_poses)
    objects[:,:3,3]*=body_scale
    # Minimum z contribution of a rotated cuboid about its own centre.
    minimum_local_height=-np.sum(np.abs(objects[:,2,:3])*np.asarray(half_extents),axis=1)
    objects[:,2,3]+=(body_scale-1)*minimum_local_height
    return dict(human_joints_robot_world=joints,nominal_object_poses=objects,
                robot_world_from_reconstruction=transform,body_scale=float(body_scale),
                box_half_extents=np.asarray(half_extents).copy())


def select_contact_targets(force_evidence,contact_arrays,metadata,policy='visual_or_necessary'):
    """One candidate patch per semantic region; retain the source uncertainty.

    Group-derived targets represent one feasible assignment, never unique
    anatomical localization. They receive a lower IK weight than visual support.
    """
    if policy not in ('visual_or_necessary','all_explanatory','hands_only'):
        raise ValueError('Unknown contact policy')
    active=contact_arrays['active'];poses=contact_arrays['world_from_object']
    if len(force_evidence)!=len(active) or len(poses)!=len(active):raise ValueError('Contact evidence frames do not align')
    groups=metadata.get('support_groups',{})
    output=[]
    for frame,record in enumerate(force_evidence):
        if record.get('frame',frame)!=frame:raise ValueError('Contact evidence was reordered')
        selected={}
        for candidate in record['candidates']:
            if candidate['source']!='human':continue
            region=candidate['region']
            if policy=='hands_only' and region not in ('left_hand','right_hand'):continue
            ids=np.asarray(candidate['vertex_ids'],int)
            visual_count=int(active[frame,ids].sum())
            visual=visual_count>=metadata['contact_config']['min_region_vertices']
            individual=bool(candidate['necessary'] and candidate['explanatory'])
            group=bool(candidate['explanatory'] and any(record.get('necessary_groups',{}).get(name,False)
                       and region in members for name,members in groups.items()))
            explanatory=bool(candidate['explanatory'])
            eligible=visual or individual or group or (policy=='all_explanatory' and explanatory)
            if not eligible:continue
            rank=(visual,individual,group,visual_count,float(np.linalg.norm(candidate['force_world'])))
            item=dict(region=region,point_object=candidate['point_object'],
                outward_normal_object=(-poses[frame,:3,:3].T@np.asarray(candidate['normal_world'])).tolist(),
                evidence=dict(visual_geometric=visual,individual_necessity=individual,group_assignment=group,
                              one_force_explanation=explanatory),
                weight=1. if visual else (.8 if individual else (.5 if group else .25)),
                assignment_unique=False)
            if region not in selected or rank>selected[region][0]:selected[region]=(rank,item)
        output.append([value[1] for value in selected.values()])
    return output
