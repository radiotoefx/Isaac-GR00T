# RMBench / RoboTTT current state

Updated: 2026-08-18

This is the current entry point for project status. Older scripts and reports are
historical evidence unless this document explicitly promotes them into the formal
pipeline.

## Authority boundary

- The current SAPIEN runtime and `RemoteGR00TPolicy` adapter run on the 4090.
- 4090 rollout telemetry is authoritative for online observation semantics.
- The RMBench checkout visible on PPU is stale for the online observation path. Do
  not change that adapter based on its `drive_target` implementation.
- PPU hosts GR00T training/inference work and the server protocol patch.

## Current semantics

| Surface | Current closed loop | New actual-state dataset/model |
| --- | --- | --- |
| arm state | actual articulation qpos | actual articulation qpos |
| gripper state | commanded normalized target | physical aperture |
| raw arm action | future commanded arm | future commanded arm |
| processed arm action | checkpoint processor subtracts its reference state | future command minus actual qpos, exactly once in the processor |
| gripper action | absolute | absolute |

The old clean60k checkpoint learned commanded-arm-state to future-commanded-arm.
Using it with current actual-arm-state input is a train/deploy distribution mismatch,
not evidence that the 4090 still sends drive targets.

## Evidence already accepted

- Swap Blocks actual-state collector: 5 episodes, 3003 frames, 5/5 expert success.
- Observe-and-pickup collector: 149 frames, expert success.
- Command sequence max error: 0.
- Arm tracking RMSE: approximately 0.0012-0.0016 rad.
- Nine-task first-40 precheck: 360/360 ready.
- PPU audit and raw historical metric locations:
  `/mnt/cpfs/fx/docs/worklog/2026-08-18_RoboTTT_PPU_Audit.md`.
- Independent OSS landing directory provisioned on PPU:
  `/mnt/oss/fx/RMBench/actual_state_v1`.

The OSS directory exists and is PPU-writable. This does not prove that the 4090
mount or credentials can write it; that check must run on the 4090.

## Formal gates

1. Verify a 4090 write/read/delete probe in `actual_state_v1`; never overwrite the
   historical `lerobot_v2.1` dataset.
2. Freeze LeRobot v2.1 commit/lock, converter commit and invocation, temporal shift,
   modality/processor configs, and stats script.
3. Save raw absolute future arm commands. Let the GR00T processor subtract actual
   arm qpos exactly once. Pass raw-HDF5 and processed ReplayPolicy framewise gates.
4. Train task-balanced actual-state Vanilla without simultaneously changing loss,
   sampling, or action representation. Run finite canary, then milestone gates.
5. Select a closed-loop-passing Vanilla parent by SHA256. Fork RoboTTT only from
   that exact parent.
6. Before any formal PPU rollout, require clean/scoped commits, complete metadata,
   a launch manifest, explicit loaded/missing/unexpected tensor lists, and an
   identity/wire smoke.
7. Compare Vanilla, RoboTTT normal, reset-every-query, and mid-reset with identical
   slow weights. Memory reset and RNG reset are separate controls.

## Server protocol

`scripts/ppu/run_gr00t_rmbench_policy_server.sh` now requires an explicit physical
PPU, policy seed, checkpoint, and new manifest path. The server exposes:

- `get_metadata`: repository/checkpoint/processor/stats identity, state/action
  semantics, noise mode, memory identity, checkpoint load report, plus explicit
  `formal_eligible/formal_blockers` gate fields;
- `reset`: `memory_cleared`, `rng_reseeded`, `policy_seed`, `noise_mode`, and
  `noise_id`;
- `get_action` info: per-query `noise_id` and RoboTTT per-layer
  `memory_step/update_norm/residual_norm/nonzero_fraction`.

`episode_common` reuses one stored initial-noise tensor. It is not implemented by
repeated seeded resets. Checkout `dirty: true` remains visible and is not formally
eligible merely because the endpoint exists.

## Explicitly excluded

- Old Vanilla-3k RoboTTT checkpoints are mechanism pilots, not formal candidates.
- Old command-state stats cannot be reused for the actual-state model.
- Historical common-noise boundary-jump numbers are single-window Vanilla-3k
  mechanism evidence only.
- No formal rollout budget is spent before metadata/manifest and conversion gates.
