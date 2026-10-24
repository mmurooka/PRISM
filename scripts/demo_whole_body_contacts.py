#!/usr/bin/env python3
"""Generate an explicitly synthetic support experiment and an explanatory video."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
from prism_reconstruction.pipeline import run_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh output directory to preserve experiment provenance')
    args.output.mkdir(parents=True, exist_ok=True)
    fps, count = 20, 120
    times = np.arange(count)/fps
    knee_truth = (times >= 2) & (times < 4)
    poses = np.tile(np.eye(4), (count,1,1))
    poses[:,0,3] = .12*np.sin(times)
    poses[:,2,3] = 1.1
    poses[:,:3,:3] = Rotation.from_euler('z', .12*np.sin(times/2)).as_matrix()
    half = np.array([.45,.32,.5])
    local = np.zeros((count,9,3))
    for t in range(count):
        hand_y = -.22 if knee_truth[t] else 0
        points = np.array([[-.3,hand_y,-.5], [.3,hand_y,-.5], [0,.3,-.5 if knee_truth[t] else -.7]])
        local[t] = np.repeat(points, 3, axis=0)
        local[t,:,0] += np.tile([-.005,0,.005],3)
    human = np.einsum('tij,tvj->tvi', poses[:,:3,:3], local) + poses[:,None,:3,3]
    scores = np.ones((count,9)); scores[:,6:] = .15
    regions = np.repeat(['left_hand','right_hand','left_knee'],3)
    source = args.output/'synthetic_input.npz'
    np.savez_compressed(source, timestamps_s=times, human_vertices_world=human,
                        human_vertex_regions=regions, world_from_object=poses,
                        box_half_extents=half, contact_scores=scores, gravity_world=[0,0,-9.81],
                        length_unit='m', world_frame='synthetic_world', object_id='synthetic_box',
                        scale_source='synthetic_known_dimensions', shape_assumption='known_synthetic_cuboid',
                        source_video_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run_archive(source, args.output/'inference')
    evidence = json.loads((args.output/'inference/force_evidence.json').read_text())
    with np.load(args.output/'inference/contacts.npz') as results:
        full = results['normalized_wrench_residual']
        hands = results['hand_only_wrench_residual']
        knee_index = list(results['region_names']).index('left_knee')
        knee = results['dynamics_necessary'][:,knee_index]
        visual_knee = results['region_vertex_counts'][:,knee_index] > 0
    summary = dict(experiment='synthetic_only', frames=count,
                   knee_interval_s=[2,4], knee_visual_prior=.15,
                   full_residual_mean=float(full.mean()), hand_only_residual_in_knee_interval=float(hands[knee_truth].mean()),
                   knee_necessary_recall=float(knee[knee_truth].mean()),
                   knee_necessary_false_positive_rate=float(knee[~knee_truth].mean()),
                   visual_knee_recall=float(visual_knee[knee_truth].mean()),
                   note='Controlled analytic experiment, not validation on the user video or a real robot.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*half
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    colors = {'left_hand':'#d54d42','right_hand':'#e49c23','left_knee':'#8257bb'}
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    video = args.output/'synthetic_support.mp4'
    writer = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgba','-s','1280x720','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    try:
        for t in range(count):
            fig.clear()
            ax = fig.add_subplot(121,projection='3d')
            world_corners = corners @ poses[t,:3,:3].T + poses[t,:3,3]
            ax.add_collection3d(Poly3DCollection([world_corners[f] for f in faces],facecolors='#80c5db',edgecolors='#327087',alpha=.3))
            for i, name in enumerate(colors):
                point = human[t,3*i:3*i+3].mean(axis=0)
                ax.scatter(*point,color=colors[name],s=90,label=name)
            for candidate in evidence[t]['candidates']:
                force = np.array(candidate['force_world'])
                point = poses[t,:3,:3] @ np.array(candidate['point_object']) + poses[t,:3,3]
                if np.linalg.norm(force)>.2:
                    ax.quiver(*point,*(force*.045),color='#14805e',arrow_length_ratio=.15)
            ax.set(xlim=(-.9,.9),ylim=(-.7,.7),zlim=(0,1.9),xlabel='World X [m]',ylabel='World Y [m]',zlabel='World Z [m]')
            ax.set_box_aspect([1.8,1.4,1.9]); ax.view_init(22,-60); ax.legend(loc='upper left',fontsize=8)
            ax.set_title(f'Synthetic support geometry | t={times[t]:.2f} s')
            plot = fig.add_subplot(222)
            plot.axvspan(2,4,color='#8257bb',alpha=.12,label='Constructed knee-support phase')
            plot.plot(times,hands,color='#cc5d45',label='Hand-only wrench residual')
            plot.plot(times,full,color='#14805e',label='Whole-body wrench residual')
            plot.axhline(.08,ls='--',color='gray',lw=1,label='Feasibility tolerance')
            plot.axvline(times[t],color='black',lw=1)
            plot.set(xlim=(0,6),ylim=(-.005,max(.2,hands.max()*1.1)),ylabel='Normalized residual')
            plot.legend(fontsize=8,loc='upper left'); plot.grid(alpha=.2)
            state = fig.add_subplot(224)
            state.step(times,knee.astype(float),where='post',color='#8257bb',label='Knee required by wrench balance')
            state.step(times,visual_knee.astype(float),where='post',color='#d54d42',ls='--',label='Visual/geometric threshold only')
            state.axvline(times[t],color='black',lw=1)
            state.set(xlim=(0,6),ylim=(-.15,1.3),xlabel='Time [s]',ylabel='Knee contact evidence')
            state.legend(fontsize=8,loc='upper right'); state.grid(alpha=.2)
            fig.suptitle('SYNTHETIC TEST — not a reconstruction of the real video',fontsize=15,fontweight='bold')
            fig.text(.52,.01,'Green arrows: explanatory forces per unit object mass.\nBody points and contact candidates are constructed for this test.',fontsize=9)
            fig.tight_layout(rect=(0,.05,1,.95))
            fig.canvas.draw()
            if t == 60:
                fig.savefig(args.output/'synthetic_support.png',dpi=100)
            writer.stdin.write(fig.canvas.buffer_rgba())
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError('video encoding failed')
    except BaseException:
        writer.kill(); writer.wait(); raise
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
