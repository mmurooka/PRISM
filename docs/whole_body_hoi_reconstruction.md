# Whole-body contact reconstruction extension

This is an independent reconstruction module added beside PRISM's published
training code. It does not reproduce an unpublished PRISM reconstruction frontend.
The current executable stage consumes aligned human geometry, cuboid motion and
visual contact scores. A real-video frontend, geometric G1 retargeting, RL, and
sim2real evaluation remain separate stages to validate.

## Research target and available resources

The target video is `move_box_adjusted_by_leg.mp4` in the sibling CRISP-Real2Sim
repository. Hands and knees may contact the moving box simultaneously. Human
height, box dimensions, mass, friction and inertia are not measured. The available
compute is one 16 GB RTX 5000 Ada Laptop GPU. The target robot is a 29 DoF G1 with
fixed hands; real hardware is available. Licensed weights and videos must stay
outside Git. Japanese numbered HTML reports are in CRISP-Real2Sim/reports (ignored).

## Implemented stage

`prism_reconstruction` provides:

- Cuboid surface distances, normals and object-local contact coordinates.
- SMPL skinning-based whole-body surface labels with explicit knee regions.
- Visual/geometric contact hysteresis, minimum duration, and sticking/sliding modes.
- Robust rigid pose fitting from segmented 3D tracks; missing estimates remain invalid.
- Newton–Euler wrench demand using time-aware local polynomial derivatives.
- Nonnegative contact forces inside polygonal friction cones, including floor support.
- Region-removal tests: whether removing all candidates of a body region makes the
  estimated motion infeasible under the assumed geometry and force model.
- A hand-only ablation and an archive of individual explanatory force candidates.

Necessary means *model-conditionally necessary*, not measured or uniquely
identified. A feasible fit alone is not evidence that its particular force allocation
occurred. Redundant contacts remain ambiguous. Gravity and scene support are included;
the object does not receive forces exclusively from the human.

```bash
../CRISP-Real2Sim/.venv/bin/python scripts/reconstruct_whole_body_contacts.py \
  --input /path/to/aligned_input.npz --output /path/to/new_result

../CRISP-Real2Sim/.venv/bin/python tests/unit/test_whole_body_reconstruction.py
../CRISP-Real2Sim/.venv/bin/python scripts/demo_whole_body_contacts.py \
  --output /path/to/new_synthetic_experiment
```

The synthetic demonstration intentionally gives the knee a low visual score and
constructs a support phase that cannot be explained by hands alone. It verifies
the computation under known conditions. It does not measure real-video accuracy.

## Input archive contract

All geometry must already share one metric world frame. Pose matrices map
object-local column vectors to world coordinates. NPZ loading disables pickle.

| Field | Shape / meaning |
| --- | --- |
| `timestamps_s` | T strictly increasing actual video timestamps |
| `human_vertices_world` | T × V × 3, metres, consistent vertex identities |
| `human_vertex_regions` | V Unicode semantic region names |
| `contact_scores` | T × V finite values in [0,1], not logits |
| `world_from_object` | T × 4 × 4, finite proper rigid transforms |
| `box_half_extents` | 3 positive lengths in metres |
| `gravity_world` | 3, gravitational acceleration in m/s² |
| `floor_normal_world`, `floor_offset_m` | Optional plane n·x = offset, supplied together |
| `inertia_per_mass`, `com_object` | Optional 3 × 3 and 3 mass-distribution parameters |
| `length_unit` | Must be `m` |
| `world_frame`, `scale_source`, `object_id` | Required provenance strings |
| `source_video_sha256` | Source identification; synthetic demos identify their generator |

Do not silently combine TAPIP3D, camera, CRISP scene or G1 coordinates. Apply and
record a calibrated similarity transform first. Reject missing frames or process
continuous segments; do not fill failed poses with identity transforms.

Outputs are `contacts.npz`, `metadata.json`, and `force_evidence.json`. They include
vertex contact states, intervals, object-local surface points, region-level
dynamic evidence, force directions/magnitudes, and the hand-only residual. Unknown
mass produces N/kg and Nm/kg; these are not measured forces. Unknown inertia defaults
to a uniform solid cuboid, an assumption especially questionable for a cardboard box.

## Limits and required research evaluation

The contact necessity test evaluates each frame independently. The visual/geometric
stage has temporal hysteresis. Optional `--temporal-force-smoothness` adds a sparse
sequence optimization of region forces inside hard friction cones; its per-frame
feasibility residuals are stored separately. Missing candidate forces stay zero;
the temporal prior must not be interpreted as contact identification. Human poses
and object poses are not jointly optimized here.
The solver also removes semantic groups together (knee plus thigh, each arm, or
trunk). If knee and thigh can replace one another, neither individual region is
identifiable, but their combined support may still be necessary. Archives retain
both individual and group-level tests; group necessity must not be reported as an
exact knee contact label. All necessity flags are false when the full fit is
infeasible.
`contact_evidence_code` retains separate per-vertex bits for visual/geometric
support, a positive force in one feasible explanation, individual-region
necessity and group necessity. `contact_candidates` is their union, not a binary
ground-truth mask. In particular, a dynamics-only candidate does not become a
positive InteractVLM prediction, and one selected force allocation does not
uniquely identify a contact point.
Unseen box faces, occluded body parts, camera motion, metric scale and numerical
accelerations can substantially alter the answer. Edge/corner normals are represented
by one deterministic adjacent face. Region surface points on each face are collapsed
to a centroid; wide distributed contacts may need several patches.

Before making research claims, compare hand-only, visual whole-body, geometry-only,
and dynamics-assisted contact variants on independently labelled video intervals.
Use uncertainty sweeps for scale, box dimensions, center of mass, inertia, friction,
depth and timing. Report ambiguous and infeasible frames as well as contact precision,
recall, temporal boundaries, pose accuracy and wrench residuals. Validate on more than
the motivating video. Contact necessity in one noiseless constructed example is not
an estimate of generalization.

## G1 / RL integration boundary

These archives are human reconstruction outputs, not PRISM's existing G1 contact
sidecars. The pinned runtime's region vocabulary and lower-body contact penalties
assume a different contact design. Mapping a human knee to a robot knee label is
insufficient: fixed-hand geometry, knee shape, joint limits, self-collision, support
balance and achievable force directions must be considered.

Retargeting should retain semantic body region, object-local contact location,
normal, temporal mode and confidence, while allowing measurable relaxation for
infeasible contacts. Geometric IK must be evaluated separately from dynamic
feasibility. RL should preserve intended contacts while still distinguishing unwanted
collisions; hand-only and full-body reward variants need controlled comparisons.
Simulation success alone is not sim2real contact reproduction evidence.

### Local geometric baseline

`scripts/retarget_whole_body_contacts.py` implements an independent MuJoCo/CLARABEL
baseline for the fixed-hand 29-DoF G1. It uses floor-aligned coordinates, leg-length
motion scaling, unchanged physical box dimensions, joint bounds, soft object-local
contact targets, observed foot-stance anchors and joint robot/object pose updates.
Quaternions are integrated in generalized-velocity coordinates. Collision queries
are supplemented by analytic cuboid distances on sampled robot surfaces and all
eight box corners. A nonlinear merit check limits collision-increasing steps.
The solver records residuals, violations and stalled iterations; QP success alone
is not convergence or dynamic feasibility. Near kinematic neighbours are excluded
from self-collision checks. The canonical robot XML is an explicit input, and is
not modified in place.

Example from the CRISP environment (all outputs must be ignored):

```bash
.venv/bin/python ../PRISM/scripts/retarget_whole_body_contacts.py \
  --human results/sequence/human.npz --object results/sequence/box.npz \
  --contacts results/sequence/contacts --robot-xml /path/to/g1_29dof.xml \
  --output results/sequence/g1_ik --policy all_explanatory
```

`--warm-start` accepts an existing matching full trajectory for a separate
collision-refinement pass. Hand-only and full-body variants use the same solver.
Robot patches persist only for estimated sticking contacts; uncertain or sliding
contacts may reassign a robot surface point. Proximal/distal thigh coordinates are
transferred from the SMPL template instead of mapping every thigh patch to the knee.

`scripts/evaluate_contact_sensitivity.py` reruns contact force inference over an
explicit friction/smoothing grid. Its selection fractions measure stability over
those assumptions, not calibrated contact probabilities. It does not account for
all shape, inertia or camera uncertainty.

The optional shape-mismatch relaxation preserves the cuboid face and allows a
reported tangential contact-point shift of at most 10 cm. Normal error is penalized
fully; tangential error is weaker inside that radius and strengthens outside it.
Outputs retain `point_object_original`, the relocated `point_object`, normal error,
and relocation magnitude. Reported contact distance is to the relocated target,
and must not be described as exact preservation of the human contact point.

### Exploratory local physics / PPO

`robot_physics.py` adds 29 bounded joint torque motors to the same geometry,
with the pinned HoloSoma G1 armatures and historical local PD gains. Both robot base and box remain
free. No root assistance, object springs or weld constraints are used during a
rollout. The 1 kg uniform box is an unmeasured simulation assumption. Near-neighbour
collision exclusions match the documented IK baseline; this is not a certified
model of the user's robot. A free-fall regression checks that the free object
obeys gravity and has no actuator.

`scripts/train_local_contact_policy.py` trains Stable-Baselines3 PPO in MuJoCo,
using random reference-phase resets and short episodes. `whole_body` and
`hands_only` variants have identical observations, dynamics and allowed contacts;
only the intended-contact reward aggregation differs. Rewards include surface-point
distance and measured simulator contact-force presence in the corresponding robot
body, in addition to pose/object tracking. This is a privileged-state, single-video
feasibility baseline, not the official PRISM teacher recipe or a vision policy.
`scripts/evaluate_local_contact_policy.py` evaluates full remaining clips from
fixed phases and saves actual simulated trajectories, failures and contact metrics.
Training returns alone are not success rates. No output is authorized for direct
robot execution by these scripts.

The default PPO action uses a fraction of each motor's effort limit divided by
its proportional gain, rather than the same angle offset for every joint.
Robot/box friction is explicitly 0.6 and floor friction 0.8 (MuJoCo takes the
maximum for equal-priority pairs). These are assumptions, not measurements.

`--teacher-checkpoint` enables a residual around the released PRISM privileged
actor. `released_teacher.py` checks the checkpoint's joint order, 178-dimensional
observation contract and PD action semantics. Object coordinates are rotated from
the reconstruction's width/height/thickness chart to the teacher's
width/depth/height chart without changing physical geometry. The observation and
actor parity script compares against the pinned upstream functions; passing it
does not establish simulator equivalence or successful policy transfer. Residual
means start at zero, and evaluations preserve actual failures.

### Global temporal refinement

`scripts/refine_robot_trajectory.py` jointly fits cubic-spline robot joints,
floating base and box poses across all frames. It penalizes acceleration,
stance drift, surface penetration, contact error and approximate support-polygon
ZMP violations. The differentiable FK is checked against MuJoCo, including an
offset hinge and its Jacobian. A fixed inner-hand proxy avoids framewise changes
among finger helper meshes. Dynamics-only, non-unique contact hypotheses use a
saturating loss; visual/geometric or individually necessary targets remain
quadratic. Hand proxy normals and contact residuals are recorded explicitly.

This remains a kinematic optimization: approximate ZMP omits centroidal angular
momentum and vertical acceleration, and soft costs do not certify torque or
contact-force feasibility. `smooth_contact_trajectory.py` is a diagnostic filter
baseline; it remeasures collisions because smoothing alone can cause penetration.
Neither output is safe for robot execution. Compare all residuals and physical
rollouts before selecting a training reference.

The current default support objective differentiates total body momentum, using
each link's principal inertia, angular momentum and vertical acceleration. It
requires the resulting center of pressure to lie inside the stance-foot polygon,
with positive vertical pressure and floor-friction capacity. This is still only
a net-wrench condition; box-floor support is omitted conservatively, derivative
boundaries are less accurate, and internal joint torques are not constrained by
this objective. `--balance-model approximate` retains the simpler baseline.
The default hand patch preserves source points; `--hand-patch inner` and
`--hand-normal-weight` expose the stronger palm assumption rather than silently
imposing it. Optional `--hand-closure-weight` penalizes the minimum cuboid SDF
over sampled surfaces of each visually supported hand. This is recomputed every
gradient iteration; a fixed closest-point plane can become invalid as the hand
moves. The default weight is zero so that experimental gap closure is explicit.
Point-distance, surface-gap and inverse-dynamics diagnostics must all be checked.

The teacher residual scale is explicit and stored with each checkpoint. Increasing
it expands the reachable target angles while retaining physical motor torque
bounds. Initial Gaussian exploration is inversely scaled to preserve the same
applied-action standard deviation. Zero-residual rollouts are identical across
scales; resuming with a different scale is rejected to avoid silently changing
an existing policy's meaning.

`scripts/audit_teacher_action_capacity.py` intersects motor limits with the
instantaneous PD torque intervals reachable from each chosen residual range,
then reruns the inverse-contact fit. This is stronger than comparing a single
non-unique force allocation to action bounds. It still assumes reference states,
reset previous actions, and the selected geometric tolerance; it is not a
closed-loop reachability proof. Residual action-rate penalties use applied
teacher-action units so changing the normalized range does not dilute them.

The `coupled` contact reward uses dense approach shaping `exp(-distance/0.08)`
and a separate product of this position score with clipped normal force (0 to
1 N). Contact elsewhere on the body cannot earn the full force-guidance reward.
This addresses a failure of the earlier additive force-presence bonus, whose
high score could coexist with large target-point error. The `additive` recipe
remains available for previous checkpoints and controlled comparisons. This is
our whole-body extension, not a literal reproduction of the paper's soft force
gate. Evaluate force and position jointly per region; aggregate reward alone
does not establish knee-contact retention.
The coupled recipe also uses per-joint mean exponential tracking and broader
root/object distance kernels. This avoids losing the entire posture signal when
a few wrist joints start far from the reference. Comparisons to the legacy
recipe therefore change the shaping package, not only one reward coefficient;
hand-only versus whole-body comparisons within a recipe remain matched.

`scripts/benchmark_contact_warp.py` is an optional backend diagnostic using the
pinned runtime's MuJoCo-Warp revision. It reports actual-model throughput,
short-step CPU/GPU differences, buffer capacities and finite-state checks.
Its fixed-target PD trajectory is not a tracking experiment. Keep the Warp
dependencies in a separate environment: changing the main MuJoCo version would
invalidate the existing CPU physics comparisons. A faster physics-only number
does not establish faster end-to-end training or equivalent contact dynamics.

### Direct teacher fine-tuning

`--policy-control teacher_absolute` initializes the PPO actor from the released
178-input ELU network and makes that actor trainable. Its first 178 observations
are the original teacher contract; the remaining 367 add contact, future-motion
and tracking information. New first-layer columns start at zero, preserving the
released deterministic actor before learning. `validate_teacher_finetune.py`
checks real-checkpoint output parity and identical physical transitions for
equivalent commands; tiny neural roundoff can still separate independent contact
rollouts. The learned action directly specifies the original teacher's PD target
coordinates, bounded by its action clip and the same physical motor limits.
This differs from the default frozen teacher plus residual policy. Do not resume
a checkpoint across these action/observation contracts. Training outputs retain
source snapshots, initialization hashes and resumed-step counts.

**PD compatibility correction:** the historical local baseline used damping
ratio 1, while the released teacher's saved D gains are twice those values.
Observation/actor parity did not test this controller mismatch. New teacher-based
training therefore defaults to the exact checkpoint P and D gains. Old checkpoints
without `pd_gain_source` retain `local` during evaluation so historical results
remain reproducible; explicit `--pd-gain-source checkpoint` evaluates controller
transfer. Actual gain arrays and controller changes on resume are recorded.
The conservative physical effort caps remain those of the IK model and are not
silently replaced by the teacher's larger hip-roll limits.

`scripts/audit_robot_contact_dynamics.py` computes required generalized forces
from the reference accelerations, mass matrix, bias and passive forces. It fits
these with 29 bounded motor torques and unilateral eight-ray friction pyramids
at MuJoCo collision points. Run both a tight gap tolerance (2 mm) and a deliberately
optimistic tolerance (2 cm). A fit at positive separation is not a physical
contact, and the audit omits velocity complementarity and force continuity.
Residuals are reported separately for the floating robot, floating object and
joint motors. Synthetic tests verify floor-force direction, missing support and
motor bounds. This necessary check cannot replace an unassisted rollout.

The optional `refine_robot_trajectory.py --contact-wrench-weight` fits nonnegative
Coulomb-pyramid forces together with the robot/box spline. It balances robot and
box centroidal wrenches separately and penalizes force times contact-point gap,
including stance feet and box-floor vertices. It saves proposed forces for audit.
This is a necessary-condition refinement, not inverse dynamics certification:
joint torque limits, no-slip and exact collision-manifold consistency still need
independent checks. In particular, uncertain source hypotheses are not converted
into measured contact labels by a successful force fit.

`train_local_contact_policy.py --robot-urdf <pinned g1_29dof.urdf>` additionally
uses the training asset's aggregated articulated inertias and effort limits.
Fixed child masses are counted once. Joint frames, collision geometry, reference
points and the simulator remain local. Source SHA and a `.profile.json` manifest
record this assumption. The previous retargeting asset has 35.525 kg robot mass;
the pinned training URDF has 33.343 kg. Correcting this discrepancy does not by
itself establish successful teacher transfer. Runs without this option retain
the old physical model for reproducibility.

`extract_image_contact_cues.py` records silhouette-boundary corroboration from
local person/box masks, projected human vertices and the existing 3D signed gap.
It requires a persistent regional cue, saves all thresholds and hashes, and
keeps the raw VLM/dynamics labels unchanged. These are image-supported candidates,
not contact ground truth: occlusion can also make pelvis/torso boundaries appear
adjacent to the box. The current implementation has no human self-occlusion depth
test, so these cues must not automatically become hard contact constraints.

An optional reset curriculum (`--focus-start-range FIRST LAST
--focus-probability P`) mixes focused phase resets with uniform full-clip resets.
It changes training-state sampling only; fixed evaluation starts and unassisted
rollout dynamics are unchanged. Record this mixture when comparing learning
runs. A focused training interval is not a contact ground-truth annotation.

`evaluate_contact_sensitivity.py --physical-oat --friction .6 --window .2`
adds one-at-a-time metric-scale (±10%), inertia multiplier (0.5/1.5), and object
COM (±5 cm along each local axis) checks. Gravity and rotation are not scaled;
translation, body vertices, box extents and floor offset scale together. Inertia
scales with length squared. These selected perturbations do not cover parameter
interactions or provide calibrated contact probabilities. Shifted COM tests use
the nominal inertia about the shifted centre, rather than claiming a measured
mass distribution. Contact metadata now records the supplied COM and inertia.

`--edge-contact-cones` in global trajectory refinement permits forces from the
sum of adjacent cuboid-face friction cones for targets within 2 mm of an edge.
This matters for a box resting on the thigh near its bottom corner: assigning
only the side-face normal can rule out direct upward support. The combined cone
is deliberately permissive and does not infer the true surface normal. Always
compare with the exact-collision inverse-force audit and physical rollouts. This
option changes proposed robot forces only, not the human contact evidence.

For the full sequence, `--temporal-force-solver clarabel` solves the same
nonnegative least-squares objective using explicit residual variables and sparse
conic constraints, without forming `A.T @ A`. It records solver status, residuals,
iterations and solve time, and has a 100-iteration budget. Small-problem agreement
with the original iterative LSMR formulation is tested. The initial full-clip
LSMR runs were stopped after exceeding 11 minutes; their interruption logs remain
local. A faster solver does not strengthen contact identifiability.

`--contact-group right_knee right_thigh --contact-group-weight 2` is an explicit
imitation-task prior motivated by the user's statement that the leg touches the
box. It does not raise the confidence of inferred contacts or add frame labels.
When an existing target in that group is active, the best alternative patch earns
`exp(-gap/0.05) * (0.25 + 0.75 * clip(normal_force/1 N,0,1))`.
The two adjacent anatomical regions are alternatives, not simultaneous mandatory
contacts. Hand contact cannot substitute for this term. Other contact terms and
all physics/observations stay unchanged. Compare this task-conditioned variant
with the same initialization, sampling and budget; do not report it as an
unsupervised recognition improvement. Evaluations also record the longest
continuous interval with force >1 N and target gap <3 cm for each region.

Geometric refinement also accepts `--contact-group right_thigh
--contact-group-weight 10`. Here the term is the mean squared distance of the
closest selected patch per active frame, normalized by 5 mm. It prevents a
task-relevant but uncertain target from being discarded by the saturating
geometric penalty. This is an explicit task prior, not extra perception evidence.
Target confidence, active frames and object contact locations are preserved.
Check penetration, acceleration, inverse-force feasibility and unassisted
rollouts before selecting a new reference; closing a geometric gap alone is not
a physical contact demonstration.

`--policy-control reference_pd` uses reference joint positions and velocities
instead of the released teacher actor. `--inverse-torque-reference gap_0.0020.npz`
optionally adds motor torques from a matching full-frame inverse-force audit.
The position-PD target is `qref + (tau_ff + Kd*vref)/Kp`; the existing bounded
motor loop implements feedforward plus position/velocity feedback. Explanatory
contact forces are never applied to the robot root or object. Reference and
physics hashes and full frame coverage must match. With a teacher checkpoint,
only its PD calibration is used in this mode, not its actor. Residual PPO starts
with zero mean and may adjust the 29 motor targets. Evaluation also accepts
`--reference-pd` for the matched baseline without inverse feedforward.

On the selected video, inverse feedforward increased the 4.79 s-start rollout
from 0.48 to 1.36 s and produced 0.10 s of right-thigh force above 1 N with target
gap below 3 cm. This remains a short transient, not successful whole-body tracking.
Encoding the same desired targets through teacher action coordinates agrees
within 6e-14 in the first physical step, but roundoff grows in contact-sensitive
rollouts (one start differed by 0.26 s in termination). Do not assert exact
long-rollout parity or robust performance from one deterministic diagnostic.

Evaluation records both continuous semantic-body force contact and the stricter
force-plus-target-location duration. It also stores each physical box contact's
object-local position and world-frame force on the box, including floor contacts.
These details are disabled during training. On 61 evaluated simulation states,
the recorded vectors balanced the box Newton equation within 3e-13 N; enabling
recording left all five deterministic trajectories bitwise unchanged. This checks
the simulated force sign and bookkeeping, not the accuracy of human force inference.

`scripts/evaluate_contact_intervention.py` branches a deterministic rollout from
the same state and disables only the selected articulated body's box collision.
It verifies every collision-mask pair, retaining floor and other-body contacts,
and records pre-intervention state equality. This is an artificial simulator
counterfactual, not a human contact label. In the 200k residual-policy diagnostic,
right-thigh contact supplied 1.69 N s upward impulse after the 0.08 s intervention
point. Removing it changed box height by about 12 mm at 0.20 s, but only 5 mm at
0.60 s as other contacts compensated. Load-bearing contact need not be necessary
for a similar object motion; this does not establish full-motion success.

Contact duration metrics are sampled at the 50 Hz control rate. A 500 Hz check
of the 200k policy reduced the 0.16 s proxy-distance/body-force interval to
0.11 s. Moreover, a nearby robot proxy and force somewhere on its articulated
body need not coincide. Evaluation now separately sums forces whose actual
object-local contact points are within 3 cm of the target. This stricter spatial
check gave 0.08 s at 50 Hz and 0.084 s at 500 Hz on that example. Keep both the
semantic-body and actual-patch metrics; do not rename one as the other.

`scripts/audit_physics_contact_sampling.py --checkpoint PATH --output DIR
--start-time 4.79 --horizon .6` records the contact forces and derived body
positions after every MuJoCo step. These describe the pre-integration state,
so timestamps account for that offset; no extra forward call is inserted.
Inactive target samples remain in the sequence, preventing separate episodes
from being joined. The script repeats the rollout without instrumentation and
requires identical positions, velocities and actions before writing the audit.
The saved archive and JSON include checkpoint/reference/physics hashes. This
is a 500 Hz simulation measurement, not a continuous-time contact guarantee.

For the paired horizon-4 study after 200k additional steps, the right-thigh
actual-patch interval at phase 4.79 was 0.04 s versus 0.10 s at 50 Hz for the
body-force and physical-patch rewards. The 500 Hz audit measured 0.036 s versus
0.062 s, with exact recording parity in both cases. The coarse metric thus
overstated persistence in the patch-reward example. Full-clip success remained
0/5 for both policies; this is a preliminary single-seed comparison.

Optional `--contact-force-location object_patch` changes force guidance to
`sum(F_normal * exp(-(contact_point_distance/0.03)^2))` on the selected body.
It retains geometric approach shaping, source evidence, observations and physics.
The default `body` mode preserves earlier experiments. Changing this setting on
resume is recorded explicitly. It is a smooth training signal, distinct from
the hard 3 cm/1 N evaluation threshold, and does not itself guarantee success.

`--temporal-force-solver clarabel_constrained` instead minimizes force variation
and force magnitude subject to a per-frame normalized wrench residual cap of
`max(0.08, independent_fit_residual)`. A `1e-6` numerical feasibility allowance is
recorded explicitly. The already-infeasible frames keep their original slack;
this does not manufacture feasibility. This option prevents a soft smoothness
objective from sacrificing many otherwise explainable motion frames. Original
contact evidence/necessity bits are still unchanged. The full-clip test retained
333 feasible frames while reducing regional force-derivative RMS from 127.67 to
45.14 N/kg/s; these are model-conditioned explanatory forces, not measurements.
