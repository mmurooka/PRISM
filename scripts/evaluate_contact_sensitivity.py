#!/usr/bin/env python3
"""Vary contact assumptions; report stability, not calibrated probabilities."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.pipeline import reconstruct_contacts


def perturb_physical_input(data,scale=1.,inertia_multiplier=1.,com_offset=(0.,0.,0.)):
    result=dict(data)
    if scale!=1:
        result['human_vertices_world']=np.asarray(data['human_vertices_world'])*scale
        poses=np.asarray(data['world_from_object']).copy();poses[:,:3,3]*=scale;result['world_from_object']=poses
        result['box_half_extents']=np.asarray(data['box_half_extents'])*scale
        if 'floor_offset_m' in data:result['floor_offset_m']=np.asarray(data['floor_offset_m'])*scale
    result['com_object']=np.asarray(data.get('com_object',np.zeros(3)))*scale+np.asarray(com_offset)
    half=result['box_half_extents']
    nominal=np.diag([(half[1]**2+half[2]**2)/3,(half[0]**2+half[2]**2)/3,(half[0]**2+half[1]**2)/3])
    result['inertia_per_mass']=(np.asarray(data['inertia_per_mass'])*scale**2 if 'inertia_per_mass' in data else nominal)*inertia_multiplier
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--friction',type=float,nargs='+',default=[.3,.6,.9])
    parser.add_argument('--window',type=float,nargs='+',default=[.12,.20,.35])
    parser.add_argument('--physical-oat',action='store_true',help='One-at-a-time scale, inertia and COM perturbations at one friction/window setting')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if any(not np.isfinite(x) or x<=0 for x in args.friction+args.window):raise ValueError('Positive finite settings required')
    if args.physical_oat and (len(args.friction)!=1 or len(args.window)!=1):raise ValueError('Physical OAT requires one friction and one derivative window')
    with np.load(args.input) as a:data=dict(a)
    physical=[(1.,1.,(0.,0.,0.))]
    if args.physical_oat:
        physical += [(scale,1.,(0.,0.,0.)) for scale in (.9,1.1)]
        physical += [(1.,value,(0.,0.,0.)) for value in (.5,1.5)]
        physical += [(1.,1.,tuple(sign*.05 if j==axis else 0. for j in range(3))) for axis in range(3) for sign in (-1,1)]
    settings=[(friction,window,*values) for friction,window in itertools.product(args.friction,args.window) for values in physical];summaries=[];start=time.time()
    possible=[];necessary=[];group=[];feasible=[];residual=[];hand=[];vertex=[]
    for friction,window,scale,inertia_multiplier,com_offset in settings:
        modified=perturb_physical_input(data,scale,inertia_multiplier,com_offset) if args.physical_oat else data
        arrays,metadata,records=reconstruct_contacts(modified,friction=friction,smoothing_window_s=window)
        feasible.append(arrays['normalized_wrench_residual']<=.08)
        residual.append(arrays['normalized_wrench_residual']);hand.append(arrays['hand_only_wrench_residual'])
        possible.append(arrays['dynamics_possible']);necessary.append(arrays['dynamics_necessary']);group.append(arrays['dynamics_group_necessary'])
        vertex.append((arrays['contact_evidence_code']&2)>0)
        summary=dict(friction=friction,derivative_window_s=window,metric_scale=scale,inertia_multiplier=inertia_multiplier,com_offset_object_m=com_offset,feasible_frames=int(feasible[-1].sum()),
            hand_only_feasible_frames=int((hand[-1]<=.08).sum()),
            necessary_group_frames=dict(zip(arrays['dynamics_group_names'].tolist(),group[-1].sum(axis=0).tolist())))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'sensitivity.npz',settings=np.asarray([s[:2] for s in settings]),
        physical_settings=np.asarray([[s[2],s[3],*s[4]] for s in settings]),
        physical_settings_columns=np.asarray(['metric_scale','inertia_multiplier','com_x_m','com_y_m','com_z_m']),timestamps_s=data['timestamps_s'],
        feasible=np.asarray(feasible),normalized_wrench_residual=np.asarray(residual),hand_only_wrench_residual=np.asarray(hand),
        dynamics_possible=np.asarray(possible),dynamics_necessary=np.asarray(necessary),dynamics_group_necessary=np.asarray(group),
        selected_vertex_fraction=np.mean(vertex,axis=0).astype(np.float32),region_names=arrays['region_names'],
        dynamics_group_names=arrays['dynamics_group_names'])
    output=dict(source_input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_video_sha256=str(data['source_video_sha256']),settings=summaries,elapsed_s=time.time()-start,
        interpretation='Fraction of tested assumptions selecting a contact, not a calibrated probability or independent validation.',
        physical_one_at_a_time=args.physical_oat,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[('One-at-a-time scale ±10%, inertia multiplier 0.5/1.5, COM axis shifts ±5 cm; interactions not tested. Shifted COM uses nominal inertia about the shifted centre.' if args.physical_oat else 'Fixed estimated geometry and uniform-solid-cuboid inertia/centre of mass.'),
            'Does not span all friction coefficients, geometric uncertainty, mass distributions or force explanations.'])
    (args.output/'metadata.json').write_text(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':main()
