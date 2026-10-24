"""Image-boundary corroboration for reconstructed 3D contact candidates.

Silhouette adjacency is not a contact detector by itself: occlusion can produce
identical image evidence. Keep these cues separate from learned contact labels,
force necessity and ground truth.
"""
import cv2
import numpy as np


def boundary_contact_cues(vertices_camera,intrinsic,person_mask,object_mask,signed_gap_m,
                          adjacency_px=3.,projection_tolerance_px=5.,person_tolerance_px=2.,
                          maximum_gap_m=.05,maximum_penetration_m=.02):
    person=np.asarray(person_mask,dtype=np.uint8);obj=np.asarray(object_mask,dtype=np.uint8)
    if person.ndim!=2 or obj.shape!=person.shape:raise ValueError('Aligned two-dimensional masks required')
    if not np.isin(person,[0,1]).all() or not np.isin(obj,[0,1]).all():raise ValueError('Binary masks required')
    h,w=person.shape;kernel=np.ones((3,3),np.uint8)
    boundary_person=person-cv2.erode(person,kernel);boundary_object=obj-cv2.erode(obj,kernel)
    def distance(mask):
        return cv2.distanceTransform(1-mask.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if mask.any() else np.full((h,w),np.hypot(h,w),np.float32)
    adjacency=(boundary_object>0)&(distance(boundary_person)<=adjacency_px)
    surface_distance=distance(adjacency);person_distance=distance(person)
    v=np.asarray(vertices_camera);projected=v@np.asarray(intrinsic).T
    valid=np.isfinite(v).all(1)&(v[:,2]>1e-6)
    uv=projected[:,:2]/np.maximum(projected[:,2:],1e-6)
    valid &= (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    xy=np.rint(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(int)
    xy[:,0]=xy[:,0].clip(0,w-1);xy[:,1]=xy[:,1].clip(0,h-1)
    distances=surface_distance[xy[:,1],xy[:,0]];human_distance=person_distance[xy[:,1],xy[:,0]]
    close=np.isfinite(signed_gap_m)&(signed_gap_m>=-maximum_penetration_m)&(signed_gap_m<=maximum_gap_m)
    image_supported=valid&(distances<=projection_tolerance_px)&(human_distance<=person_tolerance_px)
    return dict(image_supported=image_supported,corroborated_candidate=image_supported&close,
        boundary_distance_px=distances,adjacent_boundary=adjacency,projected_uv=uv,valid_projection=valid)


def persistent_region_cues(candidates,regions,minimum_vertices=3,minimum_frames=3):
    names=np.unique(regions);counts=np.stack([candidates[:,regions==name].sum(1) for name in names],axis=1)
    active=counts>=minimum_vertices
    for column in range(active.shape[1]):
        changes=np.diff(np.r_[False,active[:,column],False].astype(int))
        for first,last in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if last-first<minimum_frames:active[first:last,column]=False
    return names,counts,active
