"""Sparse temporal inverse dynamics with hard nonnegative friction-cone rays.

This regularizes explanatory forces. It does not make ambiguous contact locations
identifiable, and its residual must be checked before using the result as evidence.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from .contacts import validate_times
from .dynamics import friction_rays


def solve_force_sequence(candidates_by_frame, wrench_world, com_world, timestamps_s,
                         characteristic_length_m, smoothness=1e-4, regularization=1e-5,
                         tolerance=.08):
    times=validate_times(timestamps_s)
    wrench=np.asarray(wrench_world,float)
    centers=np.asarray(com_world,float)
    count=len(times)
    if (len(candidates_by_frame)!=count or wrench.shape!=(count,6) or centers.shape!=(count,3)
        or not np.isfinite(wrench).all() or not np.isfinite(centers).all()
        or not np.isfinite(characteristic_length_m) or characteristic_length_m<=0
        or not np.isfinite(smoothness) or smoothness<0
        or not np.isfinite(regularization) or regularization<0
        or tolerance<=0 or not np.isfinite(tolerance)):
        raise ValueError('Invalid force sequence inputs')
    if np.any(np.diff(times)>.15):
        raise ValueError('Process separate continuous segments across timestamp gaps')
    # Smooth total force by semantic region, allowing a patch to move within it.
    names=sorted({(c.source,c.region) for frame in candidates_by_frame for c in frame})
    region_index={name:i for i,name in enumerate(names)}
    blocks=[]; rays_by_frame=[]; penalties=[]; offsets=[0]
    aggregate_rows=[]; aggregate_cols=[]; aggregate_values=[]
    lengths=np.r_[np.ones(3),np.full(3,1/characteristic_length_m)]
    target=wrench*lengths
    normalization=np.maximum(np.linalg.norm(target,axis=1),1)
    for t,frame in enumerate(candidates_by_frame):
        columns=[]; frame_rays=[]
        for c in frame:
            point=np.asarray(c.point_world,float)
            if point.shape!=(3,) or not np.isfinite(point).all() or not 0<=c.prior<=1:
                raise ValueError('Invalid force candidate')
            rays=friction_rays(c.normal_world,c.friction)
            start=offsets[-1]+8*len(frame_rays)
            region=region_index[(c.source,c.region)]
            for axis in range(3):
                aggregate_rows.extend([3*(t*len(names)+region)+axis]*8)
                aggregate_cols.extend(range(start,start+8))
                aggregate_values.extend(rays[:,axis])
            columns.append(np.c_[rays,np.cross(point-centers[t],rays)].T*lengths[:,None]/normalization[t])
            frame_rays.append(rays)
            penalties.extend([1+4*(1-c.prior)]*8)
        blocks.append(sparse.csr_matrix(np.concatenate(columns,axis=1) if columns else np.zeros((6,0))))
        offsets.append(offsets[-1]+8*len(frame))
        rays_by_frame.append(frame_rays)
    total=offsets[-1]
    physical=sparse.block_diag(blocks,format='csr')
    aggregate=sparse.coo_matrix((aggregate_values,(aggregate_rows,aggregate_cols)),
                               shape=(count*len(names)*3,total)).tocsr()
    target=(target/normalization[:,None]).reshape(-1)
    if total:
        # Force magnitudes normalized to characteristic gravity scale in N/kg.
        force_scale=max(float(np.median(np.linalg.norm(wrench[:,:3],axis=1))),1)
        matrices=[physical,sparse.diags(np.sqrt(regularization*np.asarray(penalties))/force_scale)]
        rhs=[target,np.zeros(total)]
        if count>1 and names and smoothness:
            invdt=1/np.diff(times)
            derivative=sparse.diags([-invdt,invdt],[0,1],shape=(count-1,count))
            temporal=sparse.kron(derivative,sparse.eye(len(names)*3),format='csr')@aggregate
            matrices.append(np.sqrt(smoothness)*temporal/force_scale)
            rhs.append(np.zeros(temporal.shape[0]))
        fit=lsq_linear(sparse.vstack(matrices,format='csr'),np.concatenate(rhs),
                       bounds=(0,np.inf),tol=1e-8,lsmr_tol=1e-10,max_iter=300)
        if not fit.success or not np.isfinite(fit.x).all():
            raise RuntimeError(f'Temporal force optimization failed: {fit.message}')
        coefficients=fit.x
    else:
        coefficients=np.empty(0)
    residual=np.linalg.norm((physical@coefficients-target).reshape(count,6),axis=1)
    forces=[]
    for t,rays in enumerate(rays_by_frame):
        values=coefficients[offsets[t]:offsets[t+1]]
        forces.append(np.asarray([values[8*i:8*i+8]@r for i,r in enumerate(rays)]).reshape(-1,3))
    return dict(forces_world=forces,region_names=names,
                region_forces_world=(aggregate@coefficients).reshape(count,len(names),3),
                normalized_residual=residual,feasible=residual<=tolerance,
                smoothness=smoothness)
