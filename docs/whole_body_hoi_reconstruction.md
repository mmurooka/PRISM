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
