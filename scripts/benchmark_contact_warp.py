#!/usr/bin/env python3
"""Check GPU physics compatibility and throughput on the actual G1/box model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.robot_physics import motor_parameters


@wp.kernel
def pd_control(qpos:wp.array2d(dtype=float),qvel:wp.array2d(dtype=float),
               qadr:wp.array(dtype=int),vadr:wp.array(dtype=int),target:wp.array(dtype=float),
               kp:wp.array(dtype=float),kd:wp.array(dtype=float),
               lower:wp.array(dtype=float),upper:wp.array(dtype=float),ctrl:wp.array2d(dtype=float)):
    world,joint=wp.tid()
    ctrl[world,joint]=wp.clamp(kp[joint]*(target[joint]-qpos[world,qadr[joint]])-kd[joint]*qvel[world,vadr[joint]],lower[joint],upper[joint])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--physics-scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=128)
    parser.add_argument('--frame',type=int,default=143)
    parser.add_argument('--control-steps',type=int,default=100)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--solver-iterations',type=int)
    parser.add_argument('--conditional-graph',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    wp.config.kernel_cache_dir=str(args.cache.resolve());wp.init()
    model=mujoco.MjModel.from_xml_path(str(args.physics_scene));cpu=mujoco.MjData(model)
    if args.solver_iterations:
        model.opt.iterations=args.solver_iterations;model.opt.ls_iterations=args.solver_iterations
    with np.load(args.reference/'trajectory.npz') as archive:cpu.qpos[:]=archive['qpos'][args.frame]
    mujoco.mj_forward(model,cpu);motor=motor_parameters(model);target=cpu.qpos[motor['qadr']].copy()
    substeps=int(round(.02/model.opt.timestep))
    with wp.ScopedDevice('cuda:0'):
        gpu_model=mjw.put_model(model);gpu=mjw.make_data(model,nworld=args.worlds,nconmax=128,njmax=512)
        gpu_model.opt.graph_conditional=args.conditional_graph
        # The pinned upstream put_data path casts an empty equality sum to
        # float, which Warp rejects. This model has no equalities or activations;
        # initialize its state explicitly instead of modifying the upstream tree.
        if model.neq or model.na:raise ValueError('This benchmark expects no equalities or actuator activations')
        gpu.qpos.assign(np.repeat(cpu.qpos[None],args.worlds,axis=0).astype(np.float32))
        gpu.qvel.assign(np.repeat(cpu.qvel[None],args.worlds,axis=0).astype(np.float32))
        buffers={name:wp.array(value.astype(np.int32 if name in ('qadr','vadr') else np.float32)) for name,value in motor.items() if name!='names'}
        target_gpu=wp.array(target.astype(np.float32))
        def step():
            wp.launch(pd_control,dim=(args.worlds,29),inputs=[gpu.qpos,gpu.qvel,buffers['qadr'],buffers['vadr'],target_gpu,
                buffers['kp'],buffers['kd'],buffers['lower'],buffers['upper'],gpu.ctrl])
            mjw.step(gpu_model,gpu)
        started=time.perf_counter()
        with wp.ScopedCapture() as capture:
            for _ in range(substeps):step()
        compilation=time.perf_counter()-started
        wp.capture_launch(capture.graph);wp.synchronize()
        for _ in range(substeps):
            cpu.ctrl[:]=np.clip(motor['kp']*(target-cpu.qpos[motor['qadr']])-motor['kd']*cpu.qvel[motor['vadr']],motor['lower'],motor['upper'])
            mujoco.mj_step(model,cpu)
        gpu_q=gpu.qpos.numpy();gpu_v=gpu.qvel.numpy()
        difference=dict(qpos_max_abs=float(abs(cpu.qpos-gpu_q[0]).max()),qvel_max_abs=float(abs(cpu.qvel-gpu_v[0]).max()),
                        seconds=.02,cpu_time_s=float(cpu.time),gpu_time_s=float(gpu.time.numpy()[0]),
                        world_replicas_qpos_max_abs=float(abs(gpu_q-gpu_q[:1]).max()))
        box_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'hoi_box_free');box_q=int(model.jnt_qposadr[box_joint])
        difference.update(root_position_m=float(np.linalg.norm(cpu.qpos[:3]-gpu_q[0,:3])),
            box_position_m=float(np.linalg.norm(cpu.qpos[box_q:box_q+3]-gpu_q[0,box_q:box_q+3])),
            hinge_angle_max_rad=float(abs(cpu.qpos[motor['qadr']]-gpu_q[0,motor['qadr']]).max()))
        started=time.perf_counter()
        for _ in range(args.control_steps):wp.capture_launch(capture.graph)
        wp.synchronize();elapsed=time.perf_counter()-started
        finite=bool(np.isfinite(gpu.qpos.numpy()).all() and np.isfinite(gpu.qvel.numpy()).all())
        contacts=int(gpu.nacon.numpy()[0]);constraints=int(gpu.nefc.numpy().max())
    output=dict(mujoco_version=mujoco.__version__,warp_version=wp.__version__,worlds=args.worlds,
        control_steps=args.control_steps,control_dt=.02,physics_dt=float(model.opt.timestep),
        compile_seconds=compilation,rollout_seconds=elapsed,environment_transitions_per_second=args.worlds*args.control_steps/elapsed,
        solver_iterations=int(model.opt.iterations),line_search_iterations=int(model.opt.ls_iterations),conditional_graph=args.conditional_graph,
        cpu_gpu_short_step_difference=difference,finite=finite,
        final_contacts=contacts,contact_capacity=gpu.naconmax,final_max_constraints=constraints,constraint_capacity=gpu.njmax,
        physics_xml_sha256=hashlib.sha256(args.physics_scene.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Fixed-target PD benchmark, not motion tracking or task success; CPU/GPU parity measured only at 20 ms.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)


if __name__=='__main__':main()
