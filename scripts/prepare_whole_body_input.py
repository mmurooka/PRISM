#!/usr/bin/env python3
"""Join CRISP human geometry, calibrated object tracks and InteractVLM masks."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.adapters import assemble_input, load_interactvlm_contacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human',type=Path,required=True)
    parser.add_argument('--object',type=Path,required=True)
    priors=parser.add_mutually_exclusive_group(required=True)
    priors.add_argument('--contacts',type=Path)
    priors.add_argument('--geometry-only',action='store_true',
                        help='Explicit ablation: all vertices are allowed, with no learned visual prior')
    parser.add_argument('--gravity-world',type=float,nargs=3,required=True,
                        help='Gravity in the exported world frame; do not assume a camera axis is up')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    with np.load(args.human,allow_pickle=False) as h, np.load(args.object,allow_pickle=False) as o:
        if args.geometry_only:
            scores=np.ones(h['human_vertices_world'].shape[:2],dtype=np.float32)
        else:
            scores=load_interactvlm_contacts(args.contacts,h['frame_indices'],
                source_video_sha256=str(h['source_video_sha256'].item()))
        data=assemble_input(h,o,scores,args.gravity_world)
        data['contact_prior_source']='uninformative_geometry_only' if args.geometry_only else 'interactvlm'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**data)
    print(args.output)


if __name__=='__main__': main()
