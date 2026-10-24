"""Explicit local G1 actuation assumptions for simulation-only evaluation."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# Pinned HoloSoma G1 motor armatures; kp=(2*pi*10)^2*armature and
# kd=2*(2*pi*10)*armature. Keep conservative effort limits from the IK model.
ARMATURE = dict(hip_pitch=.01017752004,hip_roll=.025101925,hip_yaw=.01017752004,
    knee=.025101925,ankle_pitch=.00721945,ankle_roll=.00721945,
    waist_yaw=.01017752004,waist_roll=.00721945,waist_pitch=.00721945,
    shoulder_pitch=.003609725,shoulder_roll=.003609725,shoulder_yaw=.003609725,
    elbow=.003609725,wrist_roll=.003609725,wrist_pitch=.00425,wrist_yaw=.00425)


def motor_armature(name):
    key=name.removesuffix('_joint').removeprefix('left_').removeprefix('right_')
    return ARMATURE[key]


def prepare_physics_scene(source,output):
    """Add torque motors; retain free robot/box, gravity and physical contacts."""
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError(output)
    model=mujoco.MjModel.from_xml_path(str(source));root=ET.parse(source).getroot()
    if model.nu:raise ValueError('Expected the unactuated IK scene')
    option=root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    option.set('timestep','0.002');option.set('integrator','implicitfast')
    option.set('gravity','0 0 -9.81');option.set('iterations','50')
    actuator=ET.SubElement(root,'actuator')
    for joint in root.iter('joint'):
        name=joint.get('name')
        if not name:continue
        idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if model.jnt_type[idx]!=mujoco.mjtJoint.mjJNT_HINGE:continue
        limits=model.jnt_actfrcrange[idx]
        if not limits[0]<0<limits[1]:raise ValueError(f'Missing torque bounds: {name}')
        joint.set('armature',str(motor_armature(name)));joint.set('frictionloss','0.1')
        ET.SubElement(actuator,'motor',name=name,joint=name,ctrllimited='true',
            ctrlrange=' '.join(map(str,limits)))
    # Match the IK baseline's explicit near-neighbour exclusions. More distant
    # self-collisions remain enabled; this is not a hardware collision model.
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    def ancestry(body):
        result=[body]
        while result[-1]:result.append(int(model.body_parentid[result[-1]]))
        return result
    box=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'hoi_box')
    paths={i:ancestry(i) for i in range(1,model.nbody) if i!=box}
    for first,a in paths.items():
        for second,b in paths.items():
            if first>=second:continue
            distance=min(a.index(x)+b.index(x) for x in set(a)&set(b))
            if distance<=5:
                ET.SubElement(contact,'exclude',body1=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,first),
                    body2=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,second))
    output.parent.mkdir(parents=True,exist_ok=True);ET.ElementTree(root).write(output)
    return output


def motor_parameters(model):
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    if len(names)!=29:raise ValueError('Expected 29 torque actuators')
    joints=model.actuator_trnid[:,0];qadr=model.jnt_qposadr[joints];vadr=model.jnt_dofadr[joints]
    inertia=np.array([motor_armature(name) for name in names]);frequency=2*np.pi*10
    return dict(names=names,qadr=qadr,vadr=vadr,kp=inertia*frequency**2,kd=2*frequency*inertia,
                lower=model.actuator_ctrlrange[:,0].copy(),upper=model.actuator_ctrlrange[:,1].copy())


def reference_velocities(model,qpos,times):
    times=np.asarray(times);result=np.zeros((len(qpos),model.nv))
    for i in range(len(qpos)):
        first=max(0,i-1);last=min(len(qpos)-1,i+1)
        mujoco.mj_differentiatePos(model,result[i],float(times[last]-times[first]),qpos[first],qpos[last])
    return result
