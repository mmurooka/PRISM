"""Estimate a floor plane from explicitly selected background depth points."""
import numpy as np


def fit_support_plane(points_world, above_point_world, threshold_m=.03, trials=256, seed=0):
    points=np.asarray(points_world,float)
    above=np.asarray(above_point_world,float)
    if points.ndim!=2 or points.shape[1]!=3 or above.shape!=(3,) or not np.isfinite(above).all():
        raise ValueError('Invalid floor samples or above-plane reference')
    points=points[np.isfinite(points).all(axis=1)]
    if len(points)<100 or threshold_m<=0 or trials<1:
        raise ValueError('Insufficient plane data or invalid fit configuration')
    rng=np.random.default_rng(seed)
    best=np.zeros(len(points),bool)
    for _ in range(trials):
        a,b,c=points[rng.choice(len(points),3,replace=False)]
        normal=np.cross(b-a,c-a)
        length=np.linalg.norm(normal)
        if length<1e-9: continue
        normal/=length
        mask=np.abs(points@normal-a@normal)<threshold_m
        if mask.sum()>best.sum(): best=mask
    if best.mean()<.7:
        raise ValueError('Selected floor samples do not support one reliable plane')
    center=points[best].mean(axis=0)
    _,singular,vt=np.linalg.svd(points[best]-center,full_matrices=False)
    if singular[1]<.05*singular[0]:
        raise ValueError('Floor samples are nearly collinear')
    normal=vt[-1]
    if normal@(above-center)<0: normal=-normal
    offset=float(normal@center)
    distance=points@normal-offset
    inliers=np.abs(distance)<threshold_m
    if inliers.mean()<.7 or normal@above-offset<.2:
        raise ValueError('Plane orientation or support fit is unreliable')
    return dict(normal_world=normal,offset_m=offset,inlier_fraction=float(inliers.mean()),
                rmse_m=float(np.sqrt(np.mean(distance[inliers]**2))))
