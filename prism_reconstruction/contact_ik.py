"""Local MuJoCo contact-constrained G1 IK. No simulator stepping or actuation.

This is an independent geometric baseline, not the unpublished PRISM
reconstructor or a dynamically feasible robot controller. Object-local contact
targets are soft and preserve the upstream uncertainty in the output.
"""
from pathlib import Path
import itertools
import xml.etree.ElementTree as ET

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


JOINT_MAP = {
    0: 'pelvis', 1: 'left_hip_pitch_link', 2: 'right_hip_pitch_link',
    4: 'left_knee_link', 5: 'right_knee_link',
    7: 'left_ankle_intermediate_1_link', 8: 'right_ankle_intermediate_1_link',
    10: 'left_ankle_roll_sphere_5_link', 11: 'right_ankle_roll_sphere_5_link',
    16: 'left_shoulder_roll_link', 17: 'right_shoulder_roll_link',
    18: 'left_elbow_link', 19: 'right_elbow_link',
    20: 'left_rubber_hand_link', 21: 'right_rubber_hand_link',
}


def make_scene(robot_xml, half_extents, output):
    """Copy the model into an auditable scene, with an unscaled free cuboid."""
    robot_xml = Path(robot_xml).resolve()
    tree = ET.parse(robot_xml); root = tree.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((robot_xml.parent / compiler.get('meshdir', '.')).resolve()))
    body = ET.SubElement(root.find('worldbody'), 'body', name='hoi_box', pos='1 0 1')
    ET.SubElement(body, 'freejoint', name='hoi_box_free')
    ET.SubElement(body, 'geom', name='hoi_box_geom', type='box',
                  size=' '.join(map(str, half_extents)), mass='1', friction='0.6 0.01 0.001',
                  rgba='0.72 0.43 0.16 0.65')
    # The mass is an explicit placeholder; this scene is only used for IK.
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output)
    return output


def matrix_quaternion(rotation):
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def point_jacobian(model, data, body, local):
    point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac, None, point, body)
    return point, jac


def distance_jacobian(model, data, first, second, maximum=.10):
    segment = np.zeros(6)
    distance = mujoco.mj_geomDistance(model, data, first, second, maximum, segment)
    if distance >= maximum or abs(distance) < 1e-8:
        return float(distance), None
    # Dividing by the SIGNED distance is necessary inside an intersection.
    normal = (segment[3:] - segment[:3]) / distance
    j1 = np.zeros((3, model.nv)); j2 = np.zeros_like(j1)
    mujoco.mj_jac(model, data, j1, None, segment[:3], int(model.geom_bodyid[first]))
    mujoco.mj_jac(model, data, j2, None, segment[3:], int(model.geom_bodyid[second]))
    return float(distance), normal @ (j2 - j1)


def relaxed_contact_error(difference,normal,maximum_slide=.10,tangent_weight=.2):
    """Full normal accuracy, soft tangential relocation up to a stated radius."""
    normal=np.asarray(normal,float);normal=normal/np.linalg.norm(normal)
    projector=np.outer(normal,normal);tangent=(np.eye(3)-projector)@difference
    length=np.linalg.norm(tangent)
    scale=tangent_weight if length<=maximum_slide else 1-(1-tangent_weight)*maximum_slide/length
    derivative=projector+scale*(np.eye(3)-projector)
    if length>maximum_slide:
        direction=tangent/length
        derivative+=(1-tangent_weight)*maximum_slide/length*np.outer(direction,direction)
    return projector@difference+scale*tangent,derivative


class ContactIK:
    def __init__(self, scene_xml,contact_weight=120.):
        if not np.isfinite(contact_weight) or contact_weight<=0:raise ValueError('Positive contact weight required')
        self.contact_weight=float(contact_weight)
        self.model = m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = d = mujoco.MjData(m)
        self.box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hoi_box')
        self.box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'hoi_box_geom')
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
        self.box_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'hoi_box_free')
        self.box_qadr = int(m.jnt_qposadr[self.box_joint])
        self.box_vadr = int(m.jnt_dofadr[self.box_joint])
        self.hinges = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        if len(self.hinges) != 29 or m.nq != 43 or m.nv != 41:
            raise ValueError('Expected a floating 29-DoF G1 and one floating object')
        self.qadr = m.jnt_qposadr[self.hinges]; self.vadr = m.jnt_dofadr[self.hinges]
        self.body_ids = {j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                         for j, name in JOINT_MAP.items()}
        if min(self.body_ids.values()) < 0:
            raise ValueError('G1 skeleton links are missing')
        self.robot_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] not in (0, self.box_body)
                            and (m.geom_contype[i] or m.geom_conaffinity[i])]
        self.pairs = [(i, self.floor_geom) for i in self.robot_geoms]
        self.pairs += [(i, self.box_geom) for i in self.robot_geoms]
        self.pairs += [(self.box_geom, self.floor_geom)]
        # The source model contains tiny helper geoms and overlapping mechanical
        # links. Check separated branches, but exclude near kinematic neighbours.
        def ancestors(body):
            result = [int(body)]
            while result[-1]: result.append(int(m.body_parentid[result[-1]]))
            return result
        ancestry = {i: ancestors(m.geom_bodyid[i]) for i in self.robot_geoms}
        for position, first in enumerate(self.robot_geoms):
            for second in self.robot_geoms[position + 1:]:
                a, b = ancestry[first], ancestry[second]
                distance = min(a.index(x) + b.index(x) for x in set(a) & set(b))
                if distance > 5:
                    self.pairs.append((first, second))
        self.surface = {}
        mujoco.mj_forward(m, d)
        for geom in self.robot_geoms:
            body = int(m.geom_bodyid[geom]); kind = m.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[geom]; start = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
                vertices = m.mesh_vert[start:start + count].astype(float)
                hull = ConvexHull(vertices)
                samples = np.concatenate([vertices[hull.vertices], vertices[hull.simplices].mean(axis=1)])
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                t = np.arange(64); z = 1 - 2 * (t + .5) / 64; a = t * 2.39996323
                samples = np.array([np.sqrt(1-z*z)*np.cos(a), np.sqrt(1-z*z)*np.sin(a), z]).T * m.geom_size[geom, 0]
            else:
                continue
            if len(samples) > 1000: samples = samples[np.linspace(0, len(samples)-1, 1000).astype(int)]
            world = samples @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
            local = (world - d.xpos[body]) @ d.xmat[body].reshape(3, 3)
            self.surface.setdefault(body, []).append(local)
        self.surface = {key: np.concatenate(value) for key, value in self.surface.items()}
        self.anchors = {}
        self.foot_anchors = {}
        self.box_corners=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*m.geom_size[self.box_geom]

    def neutral(self):
        q = self.model.qpos0.copy(); q[:3] = [0, 0, .79]
        for joint in self.hinges:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
            value = .3 if 'knee' in name else (-.15 if 'hip_pitch' in name or 'ankle_pitch' in name else 0.)
            if 'elbow' in name: value = 1.0
            if 'shoulder_roll' in name: value = .15 if name.startswith('left') else -.15
            q[self.model.jnt_qposadr[joint]] = value
        return q

    def body_scale(self, human_joints):
        self.data.qpos[:] = self.model.qpos0; mujoco.mj_forward(self.model, self.data)
        robot = np.mean([np.linalg.norm(self.data.xpos[self.body_ids[h]]-self.data.xpos[self.body_ids[k]]) +
                         np.linalg.norm(self.data.xpos[self.body_ids[k]]-self.data.xpos[self.body_ids[a]])
                         for h, k, a in [(1,4,7),(2,5,8)]])
        human = np.median([np.linalg.norm(human_joints[:,h]-human_joints[:,k],axis=1) +
                           np.linalg.norm(human_joints[:,k]-human_joints[:,a],axis=1)
                           for h,k,a in [(1,4,7),(2,5,8)]])
        return float(robot/human)

    def set_object(self, q, pose):
        start = self.box_qadr
        q[start:start+3] = pose[:3,3]; q[start+3:start+7] = matrix_quaternion(pose[:3,:3])

    def semantic_bodies(self, region):
        side = region.split('_')[0]
        suffix = region[len(side)+1:]
        names = {
            'hand': ['rubber_hand_link','thumb_link','pinky_link'],
            'forearm': ['elbow_link','wrist_roll_link','wrist_pitch_link'],
            'upper_arm': ['shoulder_yaw_link'],
            'thigh': ['hip_pitch_link','hip_roll_link','hip_yaw_link'], 'knee': ['knee_link','hip_yaw_link'],
            'shin': ['knee_link'], 'foot': ['ankle_roll_link'],
        }.get(suffix, [])
        if side in ('left','right'): names = [side+'_'+name for name in names]
        elif region in ('chest','back','spine','torso'): names = ['torso_link']
        elif region=='pelvis': names = ['pelvis_contour_link']
        ids = [mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name) for name in names]
        return [i for i in ids if i in self.surface]

    def contact_anchors(self, targets, frame):
        output = []
        for target in targets:
            region = target['region']; point = np.asarray(target['point_object'])
            bodies = self.semantic_bodies(region)
            if not bodies: continue
            old = self.anchors.get(region)
            # Preserve the robot material patch during consecutive observations.
            reuse=old is not None and old['last_frame']==frame-1
            reuse=reuse and target.get('contact_mode')=='sticking'
            if reuse and 'limb_fraction' in target:
                reuse=abs(old.get('limb_fraction',target['limb_fraction'])-target['limb_fraction'])<.12
            if reuse:
                body, local = old['body'], old['local']
            else:
                desired, _ = point_jacobian(self.model,self.data,self.box_body,point)
                choices = []
                for body in bodies:
                    samples = self.surface[body]
                    if region.endswith('thigh') and 'limb_fraction' in target:
                        hip=self.body_ids[1 if region.startswith('left') else 2]
                        knee=self.body_ids[4 if region.startswith('left') else 5]
                        bone=self.data.xpos[knee]-self.data.xpos[hip]
                        world=samples@self.data.xmat[body].reshape(3,3).T+self.data.xpos[body]
                        fraction=(world-self.data.xpos[hip])@bone/(bone@bone)
                        samples=samples[np.abs(fraction-target['limb_fraction'])<.20]
                    if region.endswith('knee'):
                        # Restrict the proxy to the joint neighbourhood, not the ankle.
                        knee = self.body_ids[4 if region.startswith('left') else 5]
                        world = samples @ self.data.xmat[body].reshape(3,3).T + self.data.xpos[body]
                        samples = samples[np.linalg.norm(world-self.data.xpos[knee],axis=1)<.10]
                    if not len(samples): continue
                    world = samples @ self.data.xmat[body].reshape(3,3).T + self.data.xpos[body]
                    idx = np.argmin(np.linalg.norm(world-desired,axis=1))
                    choices.append((np.linalg.norm(world[idx]-desired),body,samples[idx]))
                if not choices: continue
                _, body, local = min(choices,key=lambda value:value[0])
            self.anchors[region] = dict(body=body,local=local,last_frame=frame,limb_fraction=target.get('limb_fraction',0))
            output.append(dict(**target,robot_body_id=int(body),robot_body=mujoco.mj_id2name(
                self.model,mujoco.mjtObj.mjOBJ_BODY,body),robot_point_local=local.tolist()))
        return output

    def box_surface_constraints(self):
        """Analytic cuboid SDF constraints on sampled collision surfaces.

        These supplement MuJoCo's convex distance queries, whose nearest feature
        may change abruptly in deep mesh intersections. Sampling is not a proof
        of nonintersection; both checks remain in the saved diagnostics.
        """
        m,d=self.model,self.data;R=d.xmat[self.box_body].reshape(3,3)
        centre=d.xpos[self.box_body];half=m.geom_size[self.box_geom]
        output=[]
        for body,samples in self.surface.items():
            world=samples@d.xmat[body].reshape(3,3).T+d.xpos[body]
            local=(world-centre)@R;q=np.abs(local)-half
            outside=np.maximum(q,0);length=np.linalg.norm(outside,axis=1)
            gap=length+np.minimum(q.max(axis=1),0)
            normal=outside/np.maximum(length[:,None],1e-12)*np.sign(local)
            inside=length<1e-9;axis=q.argmax(axis=1)
            normal[inside]=0;ids=np.flatnonzero(inside);normal[ids,axis[ids]]=np.sign(local[ids,axis[ids]])
            # Several closest features prevent a single active face from hiding
            # the other side of a limb or a corner.
            feature=axis*2+(local[np.arange(len(local)),axis]>0)
            for value in np.unique(feature[gap<.025]):
                eligible=np.flatnonzero((feature==value)&(gap<.025))
                chosen=eligible[np.argsort(gap[eligible])[:2]]
                for idx in chosen:
                    point,jrobot=point_jacobian(m,d,body,samples[idx])
                    _,jobject=point_jacobian(m,d,self.box_body,local[idx])
                    jac=(normal[idx]@R.T)@(jrobot-jobject)
                    output.append((float(gap[idx]),jac,body))
        return output

    def solve(self, joints, object_pose, targets, initial, previous, frame, iterations=12, dt=1/30,stance=(False,False)):
        m, d = self.model, self.data; q = initial.copy()
        if previous is not None:
            # A per-frame warm start may lie outside the velocity box around
            # the already corrected previous frame. Make zero increment feasible.
            q[self.qadr]=np.clip(q[self.qadr],previous[self.qadr]-6*dt,previous[self.qadr]+6*dt)
        q[self.qadr]=np.clip(q[self.qadr],m.jnt_range[self.hinges,0],m.jnt_range[self.hinges,1])
        d.qpos[:] = q; mujoco.mj_forward(m,d)
        anchors = self.contact_anchors(targets,frame)
        feet=[]
        for index,side in enumerate(('left','right')):
            if not stance[index]:continue
            body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
            old=self.foot_anchors.get(side)
            local=np.array([[.12,.03,-.035],[.12,-.03,-.035],[-.05,0,-.035]])
            if old is not None and old['last_frame']==frame-1:
                desired=old['desired']
            else:
                ankle=7+index;toe=10+index
                direction=joints[toe,:2]-joints[ankle,:2];yaw=np.arctan2(direction[1],direction[0])
                rotation=Rotation.from_euler('z',yaw).as_matrix()
                desired=local@rotation.T+joints[ankle];desired[:,2]=0.
            self.foot_anchors[side]=dict(last_frame=frame,desired=desired)
            feet.append((side,body,local,desired))
        nominal = q.copy(); self.set_object(nominal,object_pose)
        def contact_residual(target):
            robot,jrobot=point_jacobian(m,d,target['robot_body_id'],np.asarray(target['robot_point_local']))
            R=d.xmat[self.box_body].reshape(3,3);local=R.T@(robot-d.xpos[self.box_body])
            _,jobject=point_jacobian(m,d,self.box_body,local)
            error,derivative=relaxed_contact_error(local-target['point_object'],target['outward_normal_object'])
            return error,derivative@R.T@(jrobot-jobject)
        def merit(candidate):
            d.qpos[:]=candidate;mujoco.mj_forward(m,d);value=0.
            for human,body in self.body_ids.items():
                weight=18. if human in (4,5,7,8,10,11) else 10.
                value+=weight**2*np.sum((d.xpos[body]-joints[human])**2)
            difference=np.zeros(m.nv);mujoco.mj_differentiatePos(m,difference,1,candidate,nominal)
            value+=np.sum((difference[self.box_vadr:]*np.array([14.,14.,14.,7.,7.,7.]))**2)
            if previous is not None:
                mujoco.mj_differentiatePos(m,difference,1,candidate,previous)
                weights=np.ones(m.nv)*.7;weights[:3]=2;weights[self.box_vadr:self.box_vadr+3]=3
                value+=np.sum((difference*weights)**2)
            for target in anchors:
                error,_=contact_residual(target)
                value+=self.contact_weight**2*target['weight']*np.sum(error**2)
            for side,body,locals,desired in feet:
                for local,point in zip(locals,desired):
                    actual,_=point_jacobian(m,d,body,local);value+=120**2*np.sum((actual-point)**2)
            distances=[distance_jacobian(m,d,a,b,.035)[0] for a,b in self.pairs]
            distances += [distance for distance,_,_ in self.box_surface_constraints()]
            for corner in self.box_corners:
                point,_=point_jacobian(m,d,self.box_body,corner);distances.append(point[2])
            return float(value+5e6*np.sum(np.minimum(distances,0.)**2))
        status=[];termination='iteration_limit'
        for iteration in range(iterations):
            d.qpos[:] = q; mujoco.mj_forward(m,d)
            rows=[]; residuals=[]
            def residual(jac, value, weight):
                weights=np.asarray(weight).reshape(-1)
                rows.append(np.asarray(jac).reshape(-1,m.nv)*weights[:,None])
                residuals.append(np.asarray(value).reshape(-1)*weights)
            for human,body in self.body_ids.items():
                point,jac = point_jacobian(m,d,body,np.zeros(3))
                weight = 18. if human in (4,5,7,8,10,11) else 10.
                residual(jac,point-joints[human],weight)
            # Pose difference in tangent coordinates (quaternions are never added).
            difference=np.zeros(m.nv); mujoco.mj_differentiatePos(m,difference,1,q,nominal)
            obj_weight=np.array([14.,14.,14.,7.,7.,7.])
            residual(np.eye(m.nv)[self.box_vadr:],-difference[self.box_vadr:],obj_weight)
            if previous is not None:
                difference=np.zeros(m.nv); mujoco.mj_differentiatePos(m,difference,1,q,previous)
                weights=np.ones(m.nv)*.7;weights[:3]=2;weights[self.box_vadr:self.box_vadr+3]=3
                residual(np.diag(weights),-difference*weights,1.)
            for target in anchors:
                error,jac=contact_residual(target)
                residual(jac,error,self.contact_weight*np.sqrt(target['weight']))
            for side,body,locals,desired in feet:
                for local,target in zip(locals,desired):
                    point,jac=point_jacobian(m,d,body,local)
                    residual(jac,point-target,120.)
            inequalities=[]; distances=[]
            for first,second in self.pairs:
                distance,jac=distance_jacobian(m,d,first,second,.035)
                if jac is not None:
                    inequalities.append(jac);distances.append(distance)
            for distance,jac,body in self.box_surface_constraints():
                inequalities.append(jac);distances.append(distance)
            # All eight corners matter: a single closest-corner linearization
            # can rotate an opposite corner through the floor at the next step.
            for corner in self.box_corners:
                point,jac=point_jacobian(m,d,self.box_body,corner)
                inequalities.append(jac[2]);distances.append(point[2])
            A=np.concatenate(rows); r=np.concatenate(residuals)
            delta=cp.Variable(m.nv)
            bound=np.full(m.nv,.12);bound[:3]=.035;bound[self.box_vadr:self.box_vadr+3]=.035
            constraints=[delta<=bound,delta>=-bound,
                q[self.qadr]+delta[self.vadr]>=m.jnt_range[self.hinges,0],
                q[self.qadr]+delta[self.vadr]<=m.jnt_range[self.hinges,1]]
            objective=cp.sum_squares(A@delta+r)+.03*cp.sum_squares(delta)
            if previous is not None:
                constraints += [q[self.qadr]+delta[self.vadr]-previous[self.qadr]<=6*dt,
                                q[self.qadr]+delta[self.vadr]-previous[self.qadr]>=-6*dt]
            if inequalities:
                slack=cp.Variable(len(distances),nonneg=True)
                constraints += [np.asarray(distances)+np.asarray(inequalities)@delta+slack>=0.]
                objective += 5e6*cp.sum_squares(slack)
            problem=cp.Problem(cp.Minimize(objective),constraints)
            problem.solve(solver='CLARABEL')
            status.append(problem.status)
            if problem.status not in ('optimal','optimal_inaccurate') or delta.value is None:
                raise RuntimeError(f'Frame {frame} IK failed: {problem.status}')
            step=np.asarray(delta.value)
            before=merit(q);accepted=False
            for alpha in (1.,.5,.25,.125,.0625,.03125):
                trial=q.copy();mujoco.mj_integratePos(m,trial,step,alpha)
                # Exact floor projection after rotation integration.
                rotation=Rotation.from_quat(trial[self.box_qadr+3:self.box_qadr+7][[1,2,3,0]]).as_matrix()
                trial[self.box_qadr+2]=max(trial[self.box_qadr+2],float(np.abs(rotation[2])@m.geom_size[self.box_geom]))
                if merit(trial)<=before+1e-8:q=trial;accepted=True;break
            if not accepted:termination='line_search_stalled';break
            if np.linalg.norm(step)<1e-4:termination='step_tolerance';break
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        for target in anchors:
            p,_=point_jacobian(m,d,target['robot_body_id'],np.asarray(target['robot_point_local']))
            o,_=point_jacobian(m,d,self.box_body,np.asarray(target['point_object']))
            target['original_position_error_m']=float(np.linalg.norm(p-o))
            original=np.asarray(target['point_object']);R=d.xmat[self.box_body].reshape(3,3)
            local=R.T@(p-d.xpos[self.box_body]);normal=np.asarray(target['outward_normal_object']);normal/=np.linalg.norm(normal)
            difference=local-original;tangent=difference-normal*(difference@normal)
            shift=tangent*min(1.,.10/max(np.linalg.norm(tangent),1e-12))
            relocated=np.clip(original+shift,-m.geom_size[self.box_geom],m.geom_size[self.box_geom])
            target['point_object_original']=original.tolist();target['point_object']=relocated.tolist()
            target['tangential_relaxation_m']=float(np.linalg.norm(relocated-original))
            target['normal_error_m']=float(abs(difference@normal))
            target['position_error_m']=float(np.linalg.norm(local-relocated))
        collisions=[]
        for first,second in self.pairs:
            distance,_=distance_jacobian(m,d,first,second,.02)
            if distance<-.002:
                collisions.append(dict(first=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,first),
                    second=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,second),penetration_m=-distance))
        surface_penetration=max([-distance for distance,_,_ in self.box_surface_constraints()],default=0.)
        errors=[np.linalg.norm(d.xpos[body]-joints[human]) for human,body in self.body_ids.items()]
        foot_records=[]
        for side,body,locals,desired in feet:
            actual=np.array([point_jacobian(m,d,body,local)[0] for local in locals])
            foot_records.append(dict(side=side,maximum_height_error_m=float(np.abs(actual[:,2]).max()),
                maximum_anchor_error_m=float(np.linalg.norm(actual-desired,axis=1).max())))
        return q,dict(frame=frame,targets=anchors,collision_violations=collisions,
            stance_feet=foot_records,
            maximum_sampled_surface_penetration_m=max(0.,surface_penetration),
            skeleton_error_mean_m=float(np.mean(errors)),solver_status=status[-1],iterations=len(status),
            termination=termination,
            dynamically_validated=False)
