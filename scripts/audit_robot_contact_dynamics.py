#!/usr/bin/env python3
"""Audit whether a reference admits bounded joint torques and unilateral contacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.inverse_contact_audit import audit_frame
from prism_reconstruction.robot_physics import reference_velocities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--physics-scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gaps',type=float,nargs='+',default=[.002,.02])
    parser.add_argument('--stride',type=int,default=1)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.stride<1 or min(args.gaps)<0:raise ValueError('Positive stride and nonnegative gaps required')
    m=mujoco.MjModel.from_xml_path(str(args.physics_scene));d=mujoco.MjData(m)
    with np.load(args.reference/'trajectory.npz') as archive:q=archive['qpos'];times=archive['timestamps_s']
    velocities=reference_velocities(m,q,times);accelerations=np.gradient(velocities,times,axis=0,edge_order=2)
    args.output.mkdir(parents=True);summaries=[]
    for gap in args.gaps:
        # Margin changes collision detection only. Required force is computed
        # explicitly from M*qacc+bias-passive, never from mj_inverse constraints.
        m.geom_margin[:]=max(gap,1e-6);m.geom_gap[:]=0.
        records=[];residuals=[];torques=[]
        for frame in range(0,len(q),args.stride):
            result=audit_frame(m,d,q[frame],velocities[frame],accelerations[frame],gap)
            records.append(dict(frame=frame,time_s=float(times[frame]),
                **{k:result[k] for k in ('contacts','robot_root_residual','box_residual','motor_residual','solver_success')}))
            residuals.append(result['normalized_residual']);torques.append(result['torques'])
        keys=('robot_root_residual','box_residual','motor_residual')
        errors=np.array([[r[key] for key in keys] for r in records]);success=np.array([r['solver_success'] for r in records])
        summary=dict(maximum_admitted_gap_m=gap,frames=len(records),solver_success_frames=int(success.sum()),
            necessary_fit_below_5percent_frames=int(((errors.max(1)<.05)&success).sum()),
            residuals={key:dict(median=float(np.median(errors[:,i])),p95=float(np.quantile(errors[:,i],.95))) for i,key in enumerate(keys)})
        label=f'gap_{gap:.4f}';np.savez_compressed(args.output/(label+'.npz'),normalized_residual=residuals,torques=torques)
        (args.output/(label+'.json')).write_text(json.dumps(records,indent=2));summaries.append(summary)
        print(json.dumps(summary),flush=True)
    metadata=dict(schema='inverse_robot_object_contact_audit_v1',summaries=summaries,
        source_trajectory_sha256=hashlib.sha256((args.reference/'trajectory.npz').read_bytes()).hexdigest(),
        physics_xml_sha256=hashlib.sha256(args.physics_scene.read_bytes()).hexdigest(),
        code_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).resolve().parents[1]/'prism_reconstruction/inverse_contact_audit.py')},
        assumptions=['MuJoCo collision contacts with specified positive-gap tolerance; no forces outside the tolerance.',
            'Eight-ray inscribed Coulomb friction pyramids; no torsional or rolling contact moment.',
            '29 independently bounded hinge torques; floating robot and object are unactuated.',
            'No contact velocity/complementarity or force-rate constraints: optimistic necessary check only.',
            'Motor residual normalized by effort limit; root/object translation by own weight and rotation by weight times 0.3 m.',
            'Finite-difference reference velocities and accelerations; results depend on reconstructed trajectory and model.'],
        safe_for_robot_execution=False)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
