"""Join independently produced geometry only after frame/provenance checks."""
from pathlib import Path
import numpy as np
from .geometry import smpl_vertex_regions, validate_poses
from .contacts import validate_times


def scalar(archive, field):
    if field not in archive:
        raise ValueError(f'Missing required field {field}')
    return str(np.asarray(archive[field]).item())


def assemble_input(human, objects, contact_scores, gravity_world):
    for field in ('world_frame','source_video_sha256','length_unit'):
        if scalar(human,field) != scalar(objects,field):
            raise ValueError(f'Human/object {field} mismatch; calibration must be explicit')
    if scalar(human,'length_unit')!='m':
        raise ValueError('Geometry must be calibrated in metres')
    ht=validate_times(human['timestamps_s'])
    ot=validate_times(objects['timestamps_s'])
    if ht.shape != ot.shape or not np.allclose(ht,ot,atol=1e-6,rtol=0):
        raise ValueError('Human/object presentation timestamps do not match')
    vertices=np.asarray(human['human_vertices_world'])
    if vertices.ndim!=3 or vertices.shape[0]!=len(ht) or vertices.shape[2]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid human vertex sequence')
    if scalar(human,'body_topology')!='SMPL_6890_from_SMPLX_mapping' or vertices.shape[1]!=6890:
        raise ValueError('This adapter requires the exported SMPL 6890 vertex topology')
    scores=np.asarray(contact_scores)
    if scores.shape!=vertices.shape[:2] or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Contact scores must be frame-aligned SMPL probabilities/binary masks')
    poses=validate_poses(objects['world_from_object'])
    if len(poses)!=len(ht) or ('valid' in objects and not np.asarray(objects['valid']).all()):
        raise ValueError('Object tracking has failed/missing frames; select continuous valid segments')
    regions=smpl_vertex_regions(human['template_vertices'],human['skinning_weights'],human['rest_joints'])
    result=dict(timestamps_s=ht,human_vertices_world=vertices,human_vertex_regions=regions,
        contact_scores=scores,world_from_object=poses,box_half_extents=objects['box_half_extents'],
        gravity_world=np.asarray(gravity_world,float),length_unit='m',
        world_frame=scalar(human,'world_frame'),source_video_sha256=scalar(human,'source_video_sha256'),
        scale_source=scalar(human,'scale_source'),object_id=scalar(objects,'object_id'))
    for name in ('floor_normal_world','floor_offset_m','shape_assumption','inertia_per_mass','com_object'):
        if name in objects:
            result[name]=objects[name]
    return result


def load_interactvlm_contacts(folder, frame_indices, vertices=6890):
    """Use original frame indices; missing results must not be shifted or zero-filled."""
    scores=[]
    for frame in frame_indices:
        path=Path(folder)/f'{int(frame):05d}.npz'
        # The CRISP worker uses source JPEG stems, not positions in a subsample.
        if not path.is_file():
            raise FileNotFoundError(f'Missing contact result for source frame {frame}: {path}')
        with np.load(path,allow_pickle=False) as data:
            value=np.asarray(data['pred_contact_3d_smplh']).reshape(-1)
        if value.shape!=(vertices,) or not np.isfinite(value).all() or np.any((value<0)|(value>1)):
            raise ValueError(f'Invalid SMPL contact mask in {path}')
        scores.append(value)
    return np.stack(scores)
