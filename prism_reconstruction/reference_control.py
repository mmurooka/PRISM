"""Reference PD with optional motor-only inverse-dynamics feedforward."""
import hashlib
import json
from pathlib import Path
import numpy as np


def load_inverse_torques(path, reference, physics_scene, count, motors):
    path=Path(path)
    audit=json.loads((path.parent/'metadata.json').read_text())
    for key,source in [('source_trajectory_sha256',Path(reference)/'trajectory.npz'),
                       ('physics_xml_sha256',Path(physics_scene))]:
        if audit[key]!=hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError('Inverse-torque provenance mismatch: '+key)
    records=json.loads(path.with_suffix('.json').read_text())
    if [r['frame'] for r in records]!=list(range(count)):
        raise ValueError('A complete per-frame torque audit is required')
    with np.load(path,allow_pickle=False) as archive:torques=archive['torques'].copy()
    if torques.shape!=(count,motors) or not np.isfinite(torques).all():
        raise ValueError('Invalid torque reference')
    return torques


def reference_pd_target(q, velocity, kp, kd, torque):
    """Existing position PD realizes tau_ff + Kp(qref-q) + Kd(vref-v)."""
    return q+(torque+kd*velocity)/kp
