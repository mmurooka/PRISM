#!/usr/bin/env python3
"""Preserve RGB silhouette evidence separately from contact/dynamics labels."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.geometry import box_surface
from prism_reconstruction.image_contact_cues import boundary_contact_cues,persistent_region_cues

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('human','contact-input','mask-root','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    h=np.load(args.human);c=np.load(args.contact_input)
    if str(h['source_video_sha256'])!=str(c['source_video_sha256']):raise ValueError('Source video mismatch')
    if not np.array_equal(h['timestamps_s'],c['timestamps_s']):raise ValueError('Timestamp mismatch')
    if not np.array_equal(h['human_vertices_world'],c['human_vertices_world']):raise ValueError('Human geometry mismatch')
    if not np.array_equal(h['frame_indices'],np.arange(len(h['frame_indices']))):raise ValueError('Expected consecutive source frames starting at zero')
    height,width=map(int,h['scene_image_size_hw']);poses=c['world_from_object'];vertices=h['human_vertices_world']
    local=np.einsum('tvi,tij->tvj',vertices-poses[:,None,:3,3],poses[:,:3,:3]);_,_,gap=box_surface(local,c['box_half_extents'])
    support=[];distance=[];image_support=[];adjacency_counts=[];mask_hash=hashlib.sha256()
    for frame,world in enumerate(vertices):
        masks=[]
        for name in ('person','box'):
            path=args.mask_root/name/f'dyn_mask_{frame}.npz';mask_hash.update(path.read_bytes())
            mask=np.load(path)['dyn_mask'][0].astype(np.uint8)
            masks.append(cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST))
        camera=h['world_from_camera'][frame];v=(world-camera[:3,3])@camera[:3,:3]
        result=boundary_contact_cues(v,h['scene_intrinsics'],*masks,gap[frame])
        support.append(result['corroborated_candidate']);image_support.append(result['image_supported'])
        distance.append(result['boundary_distance_px']);adjacency_counts.append(int(result['adjacent_boundary'].sum()))
    support=np.asarray(support);names,counts,active=persistent_region_cues(support,c['human_vertex_regions'])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'cues.npz',corroborated_candidate=support,image_supported=image_support,
        boundary_distance_px=distance,signed_gap_m=gap,region_names=names,region_vertex_counts=counts,
        persistent_region_cue=active,timestamps_s=h['timestamps_s'],human_vertex_regions=c['human_vertex_regions'])
    summary={}
    for column,name in enumerate(names):
        changes=np.diff(np.r_[False,active[:,column],False].astype(int))
        intervals=[dict(first_frame=int(a),last_frame=int(b-1),start_s=float(h['timestamps_s'][a]),end_s=float(h['timestamps_s'][b-1])) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
        summary[str(name)]=dict(corroborated_frames=int(active[:,column].sum()),intervals=intervals)
    metadata=dict(schema='image_boundary_corroboration_v1',regions=summary,source_video_sha256=str(h['source_video_sha256']),
        source_human_sha256=hashlib.sha256(args.human.read_bytes()).hexdigest(),source_contact_input_sha256=hashlib.sha256(args.contact_input.read_bytes()).hexdigest(),
        mask_files_combined_sha256=mask_hash.hexdigest(),scene_image_size_hw=[height,width],adjacent_boundary_pixels=adjacency_counts,
        thresholds=dict(boundary_adjacency_px=3,projection_tolerance_px=5,visible_person_tolerance_px=2,gap_m=[-.02,.05],minimum_vertices=3,minimum_frames=3),
        code_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),Path(__file__).resolve().parents[1]/'prism_reconstruction/image_contact_cues.py')},
        interpretation='Corroborated candidates only. Silhouette adjacency can be caused by occlusion. No independent contact ground truth, no self-occlusion depth test, and no contact probability calibration. Original VLM and force labels are unchanged.')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
