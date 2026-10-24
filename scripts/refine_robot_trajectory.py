#!/usr/bin/env python3
"""Jointly refine all G1/object frames with contact, smoothness and support costs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import BSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contact_ik import ContactIK,point_jacobian,distance_jacobian
from prism_reconstruction.torch_robot import TorchRobot,rotation_6d,matrix_6d,acceleration,centroidal_contact_wrench
from smooth_contact_trajectory import derivative_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=500)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--control-spacing-s',type=float,default=.13)
    parser.add_argument('--balance-model',choices=('approximate','centroidal'),default='centroidal')
    parser.add_argument('--balance-weight',type=float,default=10.)
    parser.add_argument('--hand-patch',choices=('source','inner'),default='source')
    parser.add_argument('--hand-normal-weight',type=float,default=0.)
    parser.add_argument('--hand-closure-weight',type=float,default=0.)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.iterations<1:raise ValueError('Positive iteration count required')
    torch.set_num_threads(1);started=time.time();code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependency_sha={name:hashlib.sha256((Path(__file__).resolve().parents[1]/'prism_reconstruction'/name).read_bytes()).hexdigest()
        for name in ('torch_robot.py','contact_ik.py')}
    with np.load(args.input/'trajectory.npz') as a:arrays=dict(a)
    records=json.loads((args.input/'diagnostics.json').read_text());meta=json.loads((args.input/'metadata.json').read_text())
    solver=ContactIK(args.input/'scene.xml');m,d=solver.model,solver.data
    robot=TorchRobot(m,args.device,torch.float32);tensor=robot.tensor
    original=arrays['qpos'];times=arrays['timestamps_s'];dt=float(np.median(np.diff(times)));count=len(times)
    q=original.copy()
    for start in (3,solver.box_qadr+3):
        for i in range(1,count):
            if q[i-1,start:start+4]@q[i,start:start+4]<0:q[i,start:start+4]*=-1
    q=gaussian_filter1d(q,.10/dt,axis=0,mode='nearest')
    for start in (3,solver.box_qadr+3):q[:,start:start+4]/=np.linalg.norm(q[:,start:start+4],axis=1)[:,None]
    def rotation(data,start):return tensor(Rotation.from_quat(data[:,start:start+4][:,[1,2,3,0]]).as_matrix())
    if args.control_spacing_s<=0:raise ValueError('Positive spline control spacing required')
    knots=np.r_[np.repeat(times[0],4),np.arange(times[0]+args.control_spacing_s,times[-1],args.control_spacing_s),np.repeat(times[-1],4)]
    basis_array=BSpline.design_matrix(times,knots,3).toarray();basis=tensor(basis_array)
    inverse=np.linalg.pinv(basis_array)
    initial=[tensor(q[:,solver.qadr]),tensor(q[:,:3]),matrix_6d(rotation(q,3)),
             tensor(q[:,solver.box_qadr:solver.box_qadr+3]),matrix_6d(rotation(q,solver.box_qadr+3))]
    parameters=[torch.nn.Parameter(tensor(inverse@value.cpu().numpy())) for value in initial]
    def decode():return [basis@parameter for parameter in parameters]
    joint,root,root6,box,box6=decode()
    optimizer=torch.optim.Adam([{'params':[parameters[0]],'lr':.008},{'params':[parameters[1],parameters[3]],'lr':.002},
                                {'params':[parameters[2],parameters[4]],'lr':.004}])
    initial_rates=[group['lr'] for group in optimizer.param_groups]
    prior_joint=tensor(original[:,solver.qadr]);prior_root=tensor(original[:,:3]);prior_box=tensor(original[:,solver.box_qadr:solver.box_qadr+3])
    prior_Rroot=rotation(original,3);prior_Rbox=rotation(original,solver.box_qadr+3)
    half=tensor(m.geom_size[solver.box_geom]);limits=tensor(m.jnt_range[solver.hinges])
    sample_bodies=[];sample_points=[]
    for body,points in solver.surface.items():
        chosen=np.unique(np.r_[np.linspace(0,len(points)-1,min(96,len(points))).astype(int),points.argmin(axis=0),points.argmax(axis=0)])
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body)
        if any(part in name for part in ('rubber_hand','thumb','pinky')):chosen=np.arange(len(points))
        sample_bodies.extend([body]*len(chosen));sample_points.extend(points[chosen])
    sample_bodies=torch.tensor(sample_bodies,device=args.device);sample_points=tensor(np.array(sample_points))
    contact_frames=[];contact_bodies=[];contact_robot=[];contact_object=[];contact_weights=[];contact_reliable=[]
    hand_normals=[];hand_mask=[];object_normals=[]
    palm_proxies={}
    for side,sign in (('left',-1.),('right',1.)):
        body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_rubber_hand_link')
        samples=solver.surface[body]
        eligible=samples[(samples[:,0]>.025)&(samples[:,0]<.085)&(samples[:,2]>-.025)&(samples[:,2]<.035)]
        if not len(eligible):raise ValueError('No inner-hand collision surface samples')
        desired=np.array([.055,sign*.08,.005]);point=eligible[np.argmin(np.linalg.norm(eligible-desired,axis=1))]
        palm_proxies[side]=(body,point,np.array([0.,sign,0.]))
    for frame,record in enumerate(records):
        for target in record['targets']:
            is_hand=target['region'] in ('left_hand','right_hand')
            if is_hand and args.hand_patch=='inner':
                side=target['region'].split('_')[0];body,point,normal=palm_proxies[side]
                target['source_robot_body_id']=target['robot_body_id'];target['source_robot_point_local']=target['robot_point_local']
                target['robot_body_id']=body;target['robot_body']=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body)
                target['robot_point_local']=point.tolist();target['robot_normal_local']=normal.tolist()
                target['patch_definition']='Fixed inner-hand collision-surface proxy; not freely switching among finger helper points.'
            contact_frames.append(frame);contact_bodies.append(target['robot_body_id']);contact_robot.append(target['robot_point_local'])
            contact_object.append(target['point_object']);contact_weights.append(target['weight'])
            contact_reliable.append(target['evidence']['visual_geometric'] or target['evidence']['individual_necessity'])
            hand_mask.append(is_hand);hand_normals.append(target.get('robot_normal_local',[0.,0.,1.]))
            object_normals.append(target['outward_normal_object'])
    cf=torch.tensor(contact_frames,device=args.device);cb=torch.tensor(contact_bodies,device=args.device)
    cr=tensor(contact_robot);co=tensor(contact_object);cw=tensor(contact_weights)
    reliable=torch.tensor(contact_reliable,device=args.device);hand_mask=tensor(hand_mask)
    hand_normals=tensor(hand_normals);object_normals=tensor(object_normals)
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link') for side in ('left','right')]
    foot_local=tensor([[.12,.03,-.035],[.12,-.03,-.035],[-.05,-.025,-.035],[-.05,.025,-.035]])
    stance=torch.tensor(arrays['stance'],device=args.device)
    with torch.no_grad():
        ip,iR=robot.forward(joint,root,rotation_6d(root6),box,rotation_6d(box6))
        initial_feet=(torch.einsum('tsij,vj->tsvi',iR[:,feet],foot_local)+ip[:,feet,None]).cpu().numpy()
    anchors=initial_feet.copy();anchors[:,:,:,2]=0
    for side in range(2):
        active=arrays['stance'][:,side];changes=np.diff(np.r_[False,active,False].astype(int))
        for first,last in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            anchors[first:last,side,:,:2]=np.median(initial_feet[first:last,side,:,:2],axis=0)
    anchors=tensor(anchors)
    edge_frames=[];edge_a=[];edge_b=[]
    for frame in range(count):
        indices=np.concatenate([np.arange(side*4,side*4+4) for side in range(2) if arrays['stance'][frame,side]]) if arrays['stance'][frame].any() else []
        if len(indices)<3:continue
        points=initial_feet[frame].reshape(8,3)[indices,:2];hull=ConvexHull(points).vertices
        for first,last in zip(hull,np.roll(hull,-1)):
            edge_frames.append(frame);edge_a.append(indices[first]);edge_b.append(indices[last])
    ef=torch.tensor(edge_frames,device=args.device);ea=torch.tensor(edge_a,device=args.device);eb=torch.tensor(edge_b,device=args.device)
    nominal=arrays['nominal_object_poses'];on_floor=tensor((nominal[:,2,3]-np.abs(nominal[:,2,:3])@m.geom_size[solver.box_geom])<.01)
    masses=tensor(m.body_mass);inertial=tensor(m.body_ipos)
    inertia=tensor(m.body_inertia)
    inertial_rotation=tensor(Rotation.from_quat(m.body_iquat[:,[1,2,3,0]]).as_matrix())
    self_pairs=np.array([(a,b) for a,b in solver.pairs if a!=solver.floor_geom and b!=solver.floor_geom],int)
    reliable_hand_welds=[{int(m.body_weldid[t['robot_body_id']]) for t in record['targets']
        if t['region'] in ('left_hand','right_hand') and t['evidence']['visual_geometric']} for record in records]
    hand_closure_masks=[]
    sample_weld=m.body_weldid[sample_bodies.cpu().numpy()]
    for weld in sorted(set().union(*reliable_hand_welds)):
        hand_closure_masks.append((torch.tensor(sample_weld==weld,device=args.device),
            torch.tensor([weld in active for active in reliable_hand_welds],device=args.device)))
    self_plane=None
    def snapshot():
        joint,root,root6,box,box6=decode()
        result=original.copy();result[:,solver.qadr]=joint.detach().cpu().numpy();result[:,:3]=root.detach().cpu().numpy()
        result[:,3:7]=Rotation.from_matrix(rotation_6d(root6).detach().cpu().numpy()).as_quat()[:,[3,0,1,2]]
        result[:,solver.box_qadr:solver.box_qadr+3]=box.detach().cpu().numpy()
        result[:,solver.box_qadr+3:solver.box_qadr+7]=Rotation.from_matrix(rotation_6d(box6).detach().cpu().numpy()).as_quat()[:,[3,0,1,2]]
        return result
    def refresh_self_planes():
        frames=[];bodies1=[];bodies2=[];points1=[];points2=[];normals=[]
        for frame,pose in enumerate(snapshot()):
            d.qpos[:]=pose;mujoco.mj_forward(m,d)
            distances=np.linalg.norm(d.geom_xpos[self_pairs[:,0]]-d.geom_xpos[self_pairs[:,1]],axis=1)
            radius=m.geom_rbound[self_pairs].sum(axis=1)+.025
            for a,b in self_pairs[distances<radius]:
                segment=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,int(a),int(b),.025,segment)
                if distance>=.025 or abs(distance)<1e-8:continue
                first=int(m.geom_bodyid[a]);last=int(m.geom_bodyid[b])
                frames.append(frame);bodies1.append(first);bodies2.append(last)
                points1.append((segment[:3]-d.xpos[first])@d.xmat[first].reshape(3,3))
                points2.append((segment[3:]-d.xpos[last])@d.xmat[last].reshape(3,3))
                normals.append((segment[3:]-segment[:3])/distance)
        if not frames:return None
        return (torch.tensor(frames,device=args.device),torch.tensor(bodies1,device=args.device),
                torch.tensor(bodies2,device=args.device),tensor(points1),tensor(points2),tensor(normals))
    history=[];best=None;best_loss=float('inf')
    for iteration in range(args.iterations):
        if iteration%50==0:self_plane=refresh_self_planes();best_loss=float('inf')
        for group,rate in zip(optimizer.param_groups,initial_rates):group['lr']=rate*(.1+.9*(1-iteration/args.iterations))
        optimizer.zero_grad();joint,root,root6,box,box6=decode();Rroot=rotation_6d(root6);Rbox=rotation_6d(box6)
        positions,rotations=robot.forward(joint,root,Rroot,box,Rbox)
        points=torch.einsum('tvij,vj->tvi',rotations[:,sample_bodies],sample_points)+positions[:,sample_bodies]
        local=torch.einsum('tji,tvj->tvi',Rbox,points-box[:,None]);delta=local.abs()-half
        gap=torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.amax(-1).clamp_max(0)
        penetration=torch.topk(torch.relu(-gap).square(),min(32,gap.shape[1]),dim=1).values.mean()/.01**2
        ground=torch.topk(torch.relu(-points[:,:,2]).square(),min(32,gap.shape[1]),dim=1).values.mean()/.01**2
        robot_contact=torch.einsum('nij,nj->ni',rotations[cf,cb],cr)+positions[cf,cb]
        object_contact=torch.einsum('nij,nj->ni',Rbox[cf],co)+box[cf]
        squared=(robot_contact-object_contact).square().sum(-1)/.03**2
        # One feasible force allocation does not establish that every candidate
        # really touches. Saturate unsupported distant hypotheses instead of
        # bending the robot to satisfy all of them simultaneously.
        contact=(torch.where(reliable,squared,2*squared/(2+squared))*cw).sum()/cw.sum()
        robot_normal=torch.einsum('nij,nj->ni',rotations[cf,cb],hand_normals)
        object_normal=torch.einsum('nij,nj->ni',Rbox[cf],object_normals)
        normal_loss=((robot_normal+object_normal).square().sum(-1)*hand_mask).sum()/hand_mask.sum().clamp_min(1)
        foot=torch.einsum('tsij,vj->tsvi',rotations[:,feet],foot_local)+positions[:,feet,None]
        foot_loss=(((foot-anchors).square().sum(-1))*stance[:,:,None]).sum()/stance.sum().clamp_min(1)/4/.02**2
        bottom=box[:,2]-(Rbox[:,2].abs()*half).sum(-1)
        object_floor=(torch.relu(-bottom).square().mean()+(bottom.square()*on_floor).mean())/.01**2
        prior=((joint-prior_joint)/.3).square().mean()+((root-prior_root)/.15).square().mean()
        prior+=((box-prior_box)/.12).square().mean()+.5*((Rroot-prior_Rroot)/.3).square().mean()+((Rbox-prior_Rbox)/.3).square().mean()
        smooth=(acceleration(joint,dt)/20).square().mean()+(acceleration(root,dt)/3).square().mean()+(acceleration(box,dt)/3).square().mean()
        smooth+=2*torch.topk((acceleration(joint,dt)/20).square(),4,dim=1).values.mean()
        smooth+=10*torch.topk((torch.relu(torch.diff(joint,dim=0).abs()/dt-5)/2).square(),4,dim=1).values.mean()
        smooth+=.1*(torch.diff(joint,dim=0)/dt/4).square().mean()
        for R in (Rroot,Rbox):
            change=R[1:]@R[:-1].transpose(-1,-2);skew=(change-change.transpose(-1,-2))/(2*dt)
            omega=torch.stack((skew[:,2,1],skew[:,0,2],skew[:,1,0]),-1)
            smooth+=.1*(omega/2).square().mean()+.2*(torch.diff(omega,dim=0)/dt/10).square().mean()
        centers=positions+torch.einsum('tbij,bj->tbi',rotations,inertial)
        com=(centers*masses[None,:,None]).sum(1)/masses.sum()
        force_penalty=joint.sum()*0
        if args.balance_model=='centroidal':
            force,moment=centroidal_contact_wrench(centers,rotations@inertial_rotation,masses,inertia,dt)
            zmp=torch.stack((-moment[:,1],moment[:,0]),-1)/force[:,2:].clamp_min(masses.sum()*9.81*.2)
            # Feet need positive pressure and sufficient floor friction.
            force_penalty=(torch.relu(.2*masses.sum()*9.81-force[:,2])/(masses.sum()*9.81)).square().mean()
            force_penalty+=(torch.relu(torch.linalg.vector_norm(force[:,:2],dim=-1)-.8*force[:,2])/(masses.sum()*9.81)).square().mean()
        else:zmp=com[:,:2]-com[:,2:]*acceleration(com,dt)[:,:2]/9.81
        flat=foot.reshape(count,8,3);a=flat[ef,ea,:2];b=flat[ef,eb,:2];edge=b-a;offset=zmp[ef]-a
        signed=(edge[:,0]*offset[:,1]-edge[:,1]*offset[:,0])/torch.linalg.vector_norm(edge,dim=-1).clamp_min(1e-5)
        balance=(torch.relu(-signed)/.03).square().mean()+force_penalty
        self_loss=joint.sum()*0;closure_loss=joint.sum()*0
        for points_mask,frames_mask in hand_closure_masks:
            nearest=gap[:,points_mask].amin(-1)
            closure_loss+=(nearest[frames_mask]/.005).square().mean()/len(hand_closure_masks)
        if self_plane is not None:
            f,b1,b2,p1,p2,n=self_plane
            first=torch.einsum('nij,nj->ni',rotations[f,b1],p1)+positions[f,b1]
            last=torch.einsum('nij,nj->ni',rotations[f,b2],p2)+positions[f,b2]
            violations=(torch.relu(.001-((last-first)*n).sum(-1))/.01).square()
            self_loss=torch.topk(violations,min(100,len(violations))).values.mean()
        loss=prior+smooth+10*contact+args.hand_normal_weight*normal_loss+15*foot_loss+100*penetration+100*ground+30*object_floor+args.balance_weight*balance+100*self_loss+args.hand_closure_weight*closure_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite global trajectory loss')
        value=float(loss.detach())
        if value<best_loss:best_loss=value;best=[parameter.detach().clone() for parameter in parameters]
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,100.);optimizer.step()
        with torch.no_grad():parameters[0].clamp_(limits[:,0],limits[:,1])
        item=dict(iteration=iteration,total=value,prior=float(prior.detach()),smooth=float(smooth.detach()),
            contact=float(contact.detach()),foot=float(foot_loss.detach()),penetration=float(penetration.detach()),
            hand_normal=float(normal_loss.detach()),
            hand_surface_closure=float(closure_loss.detach()),
            object_floor=float(object_floor.detach()),balance=float(balance.detach()),self_collision=float(self_loss.detach()))
        history.append(item)
        if iteration%25==0:print(json.dumps(item),flush=True)
    with torch.no_grad():
        for parameter,value in zip(parameters,best):parameter.copy_(value)
    q=snapshot();errors=[];penetrations=[];region_errors={}
    for frame,pose in enumerate(q):
        d.qpos[:]=pose;mujoco.mj_forward(m,d);record=records[frame]
        record.pop('stance_feet',None);record.pop('skeleton_error_mean_m',None)
        for target in record['targets']:
            point,_=point_jacobian(m,d,target['robot_body_id'],np.asarray(target['robot_point_local']))
            obj,_=point_jacobian(m,d,solver.box_body,np.asarray(target['point_object']))
            target['position_error_m']=float(np.linalg.norm(point-obj));errors.append(target['position_error_m'])
            region_errors.setdefault(target['region'],[]).append(target['position_error_m'])
            local=(point-d.xpos[solver.box_body])@d.xmat[solver.box_body].reshape(3,3)
            original_point=np.asarray(target.get('point_object_original',target['point_object']))
            target['original_position_error_m']=float(np.linalg.norm(local-original_point))
            target['normal_error_m']=float(abs((local-original_point)@target['outward_normal_object']))
            if 'robot_normal_local' in target:
                robot_normal=d.xmat[target['robot_body_id']].reshape(3,3)@target['robot_normal_local']
                object_normal=d.xmat[solver.box_body].reshape(3,3)@target['outward_normal_object']
                target['opposing_surface_normal_cosine']=float(-robot_normal@object_normal)
        collisions=[]
        for first,last in solver.pairs:
            distance,_=distance_jacobian(m,d,first,last,.02)
            if distance<-.002:collisions.append(dict(first=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,first),
                second=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,last),penetration_m=-distance))
        record['collision_violations']=collisions;penetrations.append(max([c['penetration_m'] for c in collisions],default=0.))
        record['maximum_sampled_surface_penetration_m']=max([max(0.,-v) for v,_,_ in solver.box_surface_constraints()],default=0.)
        record.update(solver_status='global_gradient_fit',termination='fixed_iteration_budget',iterations=args.iterations)
    arrays.update(qpos=q,joint_positions=q[:,solver.qadr],object_position=q[:,solver.box_qadr:solver.box_qadr+3],
        object_quaternion_wxyz=q[:,solver.box_qadr+3:solver.box_qadr+7])
    meta.update(source_prior_trajectory_sha256=hashlib.sha256((args.input/'trajectory.npz').read_bytes()).hexdigest(),
        global_refinement=dict(iterations=args.iterations,history=history,best_final_block_loss=best_loss,
            cubic_spline_control_spacing_s=args.control_spacing_s,control_points=basis.shape[1],
            balance_model=args.balance_model,balance_weight=args.balance_weight,
            balance_assumption='Required net contact wrench from body momentum, inside stance-foot polygon; no individual-force or torque guarantee. Box-floor support is omitted conservatively.' if args.balance_model=='centroidal' else 'Approximate planar ZMP ignores centroidal angular momentum and vertical acceleration.',
            contact_targets='Retained from prior trajectory, including its documented relocation.',
            uncertain_contact_loss='Saturating penalty for dynamics-only non-unique targets; visual/individually necessary targets remain quadratic.',
            hand_patch=args.hand_patch,hand_normal_weight=args.hand_normal_weight,
            hand_surface_closure='Optional cuboid SDF gap at nearest sampled hand surface; recomputed every gradient iteration.',
            hand_closure_weight=args.hand_closure_weight,
            collision_samples_per_body=96,collision_plane_refresh_iterations=50,
            collision_planes='Near robot/object and non-adjacent self-collision pairs from MuJoCo.'),
        derivatives_before=derivative_summary(m,original,times),derivatives_after=derivative_summary(m,q,times),
        contact_error_median_m=float(np.median(errors)),contact_error_p95_m=float(np.quantile(errors,.95)),
        contact_error_by_region={name:dict(count=len(values),median_m=float(np.median(values)),p95_m=float(np.quantile(values,.95)))
            for name,values in region_errors.items()},
        maximum_penetration_m=float(max(penetrations)),frames_penetrating_1cm=int(sum(v>.01 for v in penetrations)),
        elapsed_s=time.time()-started,refinement_code_sha256=code_sha,refinement_dependency_sha256=dependency_sha,rl_ready=False,
        method='Global cubic-spline trajectory optimization with contact, collision, floor, smoothness and selected support-wrench objective.')
    meta.pop('skeleton_error_mean_m',None)
    args.output.mkdir(parents=True);np.savez_compressed(args.output/'trajectory.npz',**arrays)
    shutil.copyfile(args.input/'scene.xml',args.output/'scene.xml')
    (args.output/'diagnostics.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
