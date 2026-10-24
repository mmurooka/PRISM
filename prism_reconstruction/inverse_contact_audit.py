"""Necessary inverse-dynamics check with bounded motors and friction cones.

Contact points come from MuJoCo collision geometry. A positive admitted gap is a
geometric tolerance, not a physical force at a distance. Feasibility is therefore
only a necessary, optimistic test; it does not guarantee contact persistence.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .dynamics import friction_rays


def contact_matrix(model, data, maximum_gap):
    columns=[];records=[]
    for contact in data.contact:
        if contact.dist>maximum_gap:continue
        first=int(model.geom_bodyid[contact.geom1]);second=int(model.geom_bodyid[contact.geom2])
        if model.body_weldid[first]==model.body_weldid[second]:continue
        jac1=np.zeros((3,model.nv));jac2=np.zeros_like(jac1)
        mujoco.mj_jac(model,data,jac1,None,contact.pos,first)
        mujoco.mj_jac(model,data,jac2,None,contact.pos,second)
        # MuJoCo contact normal points from geom1 towards geom2. Positive
        # pressure pushes body2 along this normal and body1 in the opposite way.
        rays=friction_rays(contact.frame[:3],float(min(contact.friction[:2])))
        columns.append((jac2-jac1).T@rays.T)
        records.append(dict(first_body=first,second_body=second,gap_m=float(contact.dist),
            point_world=contact.pos.tolist(),normal_world=contact.frame[:3].tolist()))
    return np.concatenate(columns,axis=1) if columns else np.zeros((model.nv,0)),records


def fit_required_force(required, contact_forces, motor_matrix, lower, upper, scale):
    """Fit forces with unilateral pyramids and signed bounded motor torques."""
    matrix=np.column_stack((contact_forces,motor_matrix));nforce=contact_forces.shape[1]
    bounds=(np.r_[np.zeros(nforce),lower],np.r_[np.full(nforce,np.inf),upper])
    weighted=matrix/np.asarray(scale)[:,None];target=required/scale
    # Tiny selection regularizer only resolves redundant force allocations.
    augmented=np.vstack((weighted,1e-7*np.eye(matrix.shape[1])))
    answer=lsq_linear(augmented,np.r_[target,np.zeros(matrix.shape[1])],bounds=bounds,
                     method='bvls',tol=1e-10,max_iter=500)
    residual=matrix@answer.x-required
    return dict(residual=residual,normalized_residual=residual/scale,
        torques=answer.x[nforce:],force_coefficients=answer.x[:nforce],
        solver_success=bool(answer.success),iterations=answer.nit)


def audit_frame(model,data,qpos,qvel,qacc,maximum_gap):
    data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(model,data)
    mass=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,mass,data.qM)
    required=mass@qacc+data.qfrc_bias-data.qfrc_passive
    matrix,contacts=contact_matrix(model,data,maximum_gap)
    box=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'hoi_box')
    box_joint=model.body_jntadr[box];box_v=int(model.jnt_dofadr[box_joint])
    motor=np.zeros((model.nv,model.nu))
    motor_v=model.jnt_dofadr[model.actuator_trnid[:,0]]
    motor[motor_v,np.arange(model.nu)]=1.
    robot_weight=(model.body_mass.sum()-model.body_mass[box])*9.81
    box_weight=model.body_mass[box]*9.81
    scale=np.ones(model.nv);scale[:3]=robot_weight;scale[3:6]=robot_weight*.3
    scale[box_v:box_v+3]=box_weight;scale[box_v+3:box_v+6]=box_weight*.3
    scale[motor_v]=np.maximum(abs(model.actuator_ctrlrange).max(1),1.)
    result=fit_required_force(required,matrix,motor,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1],scale)
    result.update(required=required,contacts=contacts,robot_root_residual=float(np.linalg.norm(result['normalized_residual'][:6])),
        box_residual=float(np.linalg.norm(result['normalized_residual'][box_v:box_v+6])),
        motor_residual=float(np.max(abs(result['normalized_residual'][motor_v]))))
    return result
