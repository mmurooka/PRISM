#!/usr/bin/env python3
"""Vary friction and motion smoothing; report stability, not probabilities."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--friction',type=float,nargs='+',default=[.3,.6,.9])
    parser.add_argument('--window',type=float,nargs='+',default=[.12,.20,.35])
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if any(not np.isfinite(x) or x<=0 for x in args.friction+args.window):raise ValueError('Positive finite settings required')
    with np.load(args.input) as a:data=dict(a)
    settings=list(itertools.product(args.friction,args.window));summaries=[];start=time.time()
    possible=[];necessary=[];group=[];feasible=[];residual=[];hand=[];vertex=[]
    for friction,window in settings:
        arrays,metadata,records=reconstruct_contacts(data,friction=friction,smoothing_window_s=window)
        feasible.append(arrays['normalized_wrench_residual']<=.08)
        residual.append(arrays['normalized_wrench_residual']);hand.append(arrays['hand_only_wrench_residual'])
        possible.append(arrays['dynamics_possible']);necessary.append(arrays['dynamics_necessary']);group.append(arrays['dynamics_group_necessary'])
        vertex.append((arrays['contact_evidence_code']&2)>0)
        summary=dict(friction=friction,derivative_window_s=window,feasible_frames=int(feasible[-1].sum()),
            hand_only_feasible_frames=int((hand[-1]<=.08).sum()),
            necessary_group_frames=dict(zip(arrays['dynamics_group_names'].tolist(),group[-1].sum(axis=0).tolist())))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'sensitivity.npz',settings=np.asarray(settings),timestamps_s=data['timestamps_s'],
        feasible=np.asarray(feasible),normalized_wrench_residual=np.asarray(residual),hand_only_wrench_residual=np.asarray(hand),
        dynamics_possible=np.asarray(possible),dynamics_necessary=np.asarray(necessary),dynamics_group_necessary=np.asarray(group),
        selected_vertex_fraction=np.mean(vertex,axis=0).astype(np.float32),region_names=arrays['region_names'],
        dynamics_group_names=arrays['dynamics_group_names'])
    output=dict(source_input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_video_sha256=str(data['source_video_sha256']),settings=summaries,elapsed_s=time.time()-start,
        interpretation='Fraction of tested assumptions selecting a contact, not a calibrated probability or independent validation.',
        limitations=['Fixed estimated geometry and uniform-solid-cuboid inertia/centre of mass.',
            'Does not span all friction coefficients, geometric uncertainty, mass distributions or force explanations.'])
    (args.output/'metadata.json').write_text(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':main()
