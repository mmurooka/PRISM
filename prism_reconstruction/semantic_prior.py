"""Provenance-checked union of binary InteractVLM and semantic cue proposals."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def augment_binary_prior(data, semantic, semantic_metadata, source_sha256):
    if str(np.asarray(data['contact_prior_source']).item()) != 'interactvlm':
        raise ValueError('The base must be the unmodified InteractVLM prior')
    if not semantic_metadata.get('complete'):
        raise ValueError('Semantic cue fusion is incomplete')
    if semantic_metadata['source_contact_input_sha256'] != source_sha256:
        raise ValueError('Semantic cues were computed for a different contact input')
    if semantic_metadata['source_video_sha256'] != str(np.asarray(data['source_video_sha256']).item()):
        raise ValueError('Semantic/base video mismatch')
    if not np.array_equal(semantic['human_vertex_regions'], data['human_vertex_regions']):
        raise ValueError('Semantic/base vertex indexing mismatch')
    if not np.array_equal(semantic['timestamps_s'], data['timestamps_s']):
        raise ValueError('Semantic/base timestamps mismatch')
    base = np.asarray(data['contact_scores'])
    candidate = np.asarray(semantic['fused_vertex_candidate'])
    if candidate.dtype != np.bool_ or candidate.shape != base.shape:
        raise ValueError('Semantic mask must match the binary base prior')
    if not np.isin(base, [0, 1]).all():
        raise ValueError('This union requires binary masks, not calibrated or continuous scores')
    output = dict(data)
    output['contact_scores_interactvlm'] = base.copy()
    output['contact_scores'] = np.maximum(base, candidate.astype(base.dtype))
    output['contact_prior_evidence_code'] = (base.astype(np.uint8) | (candidate.astype(np.uint8)*2))
    output['contact_prior_source'] = 'interactvlm_plus_local_language_image_geometry'
    output['contact_prior_score_interpretation'] = 'Binary candidate union, not calibrated probabilities or learned semantic vertex segmentation.'
    output['contact_prior_base_input_sha256'] = source_sha256
    return output


def augment_archive(source, cues, output):
    source, cues, output = Path(source), Path(cues), Path(output)
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    with np.load(source, allow_pickle=False) as data, np.load(cues/'cues.npz', allow_pickle=False) as semantic:
        source_data, semantic_data = dict(data), dict(semantic)
    cue_meta = json.loads((cues/'metadata.json').read_text())
    data = augment_binary_prior(source_data, semantic_data, cue_meta, sha(source))
    data['contact_prior_semantic_cues_sha256'] = sha(cues/'cues.npz')
    data['contact_prior_semantic_metadata_sha256'] = sha(cues/'metadata.json')
    additional = (data['contact_prior_evidence_code'] == 2)
    regions = data['human_vertex_regions']
    metadata = dict(complete=True, input=str(source.resolve()), input_sha256=sha(source),
        semantic_cues=str(cues.resolve()), semantic_cues_sha256=sha(cues/'cues.npz'),
        semantic_metadata_sha256=sha(cues/'metadata.json'),
        code_sha256=sha(__file__), additional_vertex_frames=int(additional.sum()),
        frames_with_additions=int(additional.any(axis=1).sum()),
        additional_vertices_by_region={str(name):int(additional[:, regions == name].sum()) for name in np.unique(regions)},
        prior_evidence_bits={'1':'original binary InteractVLM proposal', '2':'local language/image/geometry proposal'},
        interpretation='Separate augmented input. Original masks and original reconstruction results remain unchanged. The contact/dynamics pipeline must be rerun; force patch centroids can change when their prior weights change. No ground-truth or probability claim.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(output)
    sidecar.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return metadata
