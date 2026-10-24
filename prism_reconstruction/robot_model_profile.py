"""Transfer articulated inertia groups from a pinned robot URDF.

Fixed-link inertia is aggregated once in each moving link frame. Visual and
collision bodies remain intact. This does not equate collision geometry or the
simulator backend with the teacher's training environment.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco


def origin_transform(origin):
    result=np.eye(4)
    if origin is not None:
        result[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        result[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return result


def aggregate_urdf_inertia(path):
    root=ET.parse(path).getroot();links={x.get('name'):x for x in root.findall('link')}
    joints=root.findall('joint');children={j.find('child').get('link') for j in joints}
    bases=set(links)-children
    if len(bases)!=1:raise ValueError('A single robot tree is required')
    base=bases.pop();transforms={base:np.eye(4)};groups={base:base};pending=joints.copy()
    while pending:
        previous=len(pending)
        for joint in pending.copy():
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in transforms:continue
            if joint.get('type') not in ('fixed','revolute','continuous'):raise ValueError('Unsupported robot joint')
            transforms[child]=transforms[parent]@origin_transform(joint.find('origin'))
            groups[child]=groups[parent] if joint.get('type')=='fixed' else child;pending.remove(joint)
        if len(pending)==previous:raise ValueError('Disconnected or cyclic robot tree')
    pieces={}
    for name,link in links.items():
        entry=link.find('inertial')
        if entry is None:continue
        mass=float(entry.find('mass').get('value'))
        if mass<=0:continue
        anchor=groups[name];transform=np.linalg.inv(transforms[anchor])@transforms[name]@origin_transform(entry.find('origin'))
        a=entry.find('inertia').attrib
        inertia=np.array([[float(a[k]) for k in row] for row in [('ixx','ixy','ixz'),('ixy','iyy','iyz'),('ixz','iyz','izz')]])
        pieces.setdefault(anchor,[]).append((mass,transform[:3,3],transform[:3,:3]@inertia@transform[:3,:3].T))
    result={}
    for name,items in pieces.items():
        mass=sum(x[0] for x in items);com=sum(x[0]*x[1] for x in items)/mass
        inertia=sum(I+w*((p-com)@(p-com)*np.eye(3)-np.outer(p-com,p-com)) for w,p,I in items)
        if np.linalg.eigvalsh(inertia).min()<=0:raise ValueError('Nonpositive aggregate inertia')
        result[name]=dict(mass=mass,com=com,inertia=inertia)
    limits={j.get('name'):float(j.find('limit').get('effort')) for j in joints if j.get('type')!='fixed'}
    return result,limits


def apply_urdf_profile(root,model,path):
    properties,limits=aggregate_urdf_inertia(path)
    bodies={body.get('name'):body for body in root.iter('body')};used=set();changes=[]
    for name,prop in properties.items():
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if body<0:raise ValueError(f'Inertia anchor missing: {name}')
        weld=int(model.body_weldid[body])
        if weld!=body or weld in used:raise ValueError('Articulated groups do not match')
        used.add(weld);members=np.flatnonzero(model.body_weldid==weld)
        before=float(model.body_mass[members].sum())
        for index in members:
            xml=bodies[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(index))]
            for old in xml.findall('inertial'):xml.remove(old)
            if index!=body:
                ET.SubElement(xml,'inertial',pos='0 0 0',mass='0',diaginertia='0 0 0');continue
            I=prop['inertia'];values=[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]
            ET.SubElement(xml,'inertial',pos=' '.join(map(str,prop['com'])),mass=str(prop['mass']),fullinertia=' '.join(map(str,values)))
        changes.append(dict(body=name,previous_mass_kg=before,source_mass_kg=prop['mass']))
    actual={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE}
    if actual!=set(limits):raise ValueError('Source and target hinge vocabulary differs')
    for joint in root.iter('joint'):
        name=joint.get('name')
        if name in limits:
            joint.set('actuatorfrcrange',f'{-limits[name]} {limits[name]}');joint.set('frictionloss','0')
    for actuator in root.find('actuator'):
        bound=limits[actuator.get('joint')];actuator.set('ctrlrange',f'{-bound} {bound}')
    return dict(source_urdf=str(Path(path).resolve()),source_urdf_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        inertia_groups=changes,source_total_mass_kg=sum(x['mass'] for x in properties.values()),
        joint_frictionloss=0.,effort_limits=limits,
        limitations='Transfers URDF inertial groups and effort bounds only. Retains local joint frames and collision geometry; not exact Isaac Sim parity or a measured hardware calibration.')
