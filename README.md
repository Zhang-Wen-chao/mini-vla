# mini-vla

Teaching pipeline: train a **VLA (Vision-Language-Action) model end-to-end on PushT** —
data, flow-matching action head, distributed training, distillation, quantization, realtime serving.
The sixth repo in the mini-* series (mini-megatron / mini-deepspeed /
torch-distributed-internals / mini-vllm / mini-sglang / mini-diffusion).

**Positioning**: reuse the mechanisms I already hand-wrote in the mini-* series and
wire them into one complete VLA story: the flow-matching objective comes from
mini-diffusion, the parallelism primitives from mini-megatron / torch-distributed-internals,
ZeRO from mini-deepspeed, the serving loop from mini-vllm, and quantization from the
AWQ workflow I ran in production.

This is a **teaching project, not a production VLA**. Correctness and a complete,
traceable pipeline matter more than SOTA success rate.

## What the pipeline looks like

```
PushT dataset (zarr: 96x96 RGB obs + 2D end-effector actions)
  └─ Phase 1: flow-matching action head (π0-style objective, reused from mini-diffusion)
       v = a1 - a0   on   x_t = (1-t)*a0 + t*a1 ,  conditioned on image obs
  └─ Phase 2: distributed training (TP from mini-megatron, FSDP/HSDP from
       torch-distributed-internals, ZeRO from mini-deepspeed) + equivalence checks
  └─ Phase 3: post-training (instruction SFT, DMD-style distillation to few steps,
       AWQ-style quantization) with action-error ablation
  └─ Phase 4: realtime serving loop (mini-vllm style) with TTFT / inference-rate numbers
```

## Status

- [x] Phase 0 — repo skeleton, PushT zarr loader + sequence sampling + metrics (this repo state)
- [ ] Phase 1 — single-GPU flow-matching action head, rollout on PushT
- [ ] Phase 2 — TP / FSDP / ZeRO with equivalence verification
- [ ] Phase 3 — SFT, distillation, quantization
- [ ] Phase 4 — realtime serving loop

Full plan: [docs/plan.md](docs/plan.md). Phase 0 data notes: [docs/phase0/data-notes.md](docs/phase0/data-notes.md).

## Quickstart

```bash
# download + extract the official PushT zarr (~1.5 GB)
bash scripts/download_pusht.sh

# inspect the dataset (shapes, episode count, normalization stats)
python -m mini_vla.data --data-dir data/pusht/pusht_cchi_v7_replay.zarr

# CPU unit tests (no GPU, no dataset download needed — synthetic zarr in tmp)
pytest -q
```

### Phase 1: train a flow-matching action head

The action head conditions a CNN plus the PushT-native agent XY proprioception
on two observations and learns the
straight flow ``x_t=(1-t)a0+t*a1`` with velocity target ``a1-a0``. The CNN
keeps a 4x4 spatial feature grid before conditioning the action head, since
PushT requires mapping image locations to absolute action coordinates. Padded
action tails are excluded from its loss. It reports held-out velocity loss and
sampled normalized-action L1/L2 once per epoch, and writes a checkpoint with
the action normalization statistics. It writes both ``last.pt`` and a numbered
``epoch_XXXX.pt`` checkpoint after every epoch so long GPU runs are auditable.

```bash
python -m mini_vla.train \
  --data-dir data/pusht/pusht_cchi_v7_replay.zarr \
  --output-dir outputs/phase1 --epochs 50 --batch-size 128 --sample-steps 8 \
  --obs-horizon 2
```

For a CPU wiring smoke test, use a small synthetic zarr and pass ``--device
cpu --num-workers 0``. GPU results and simulator rollout coverage are recorded
in the Phase 1 experiment notes once the L20 run completes.

### Phase 1: rollout coverage (optional evaluator)

The core package does not depend on a simulator. Install the pinned optional
extra to evaluate the learned policy in the headless PushT physics environment
(the official task's coverage definition: T/goal intersection area divided by
goal area; success is coverage > 0.95):

```bash
pip install -e ".[rollout]"
python -m mini_vla.rollout \
  --checkpoint outputs/phase1/last.pt --episodes 20 --seed 10000 \
  --execute-steps 8 --sample-steps 8 --legacy \
  --output outputs/phase1/rollout.json
```

Each inference generates a 16-action chunk and executes its first eight actions
before re-observing. The JSON result records the mean maximum coverage, success
rate, environment seeds, action execution horizon, Euler steps, and state
assignment mode. The official image-PushT benchmark uses `legacy=true`, which is
also this evaluator's default.

### Phase 1 evidence so far

Phase 1 has 28 focused tests covering data boundaries, flow loss, sampling, checkpoints,
calibration, and rollout semantics. Before interpreting policy results, the headless evaluator
was calibrated against replay: 99.61% of 1,024 sampled recorded actions move the agent closer
to their absolute XY target; 128 one-step replay comparisons give mean agent/block position
errors of 2.45/0.86 pixels; an exact goal placement has coverage 1.0.

All policy conclusions use the official-compatible legacy=true evaluator, 20 fixed seeds, 200
control steps, and execute_steps=8. The table below is evidence, not a task-success claim:

| observation horizon | action horizon 16 | action horizon 8 |
| ---: | ---: | ---: |
| 2 | coverage 0.2674, success 0 | coverage 0.2098, success 0 |
| 4 | coverage 0.1587, success 0 | coverage 0.2185, success 0 |

The result shows an interaction between observation and action horizon, and it shows that lower
offline action L1/L2 does not guarantee a better closed-loop policy. EMA (coverage 0.1207) and
cosine learning-rate decay (0.2207) did not produce a successful trajectory. The final controlled
duration experiment (To=4/Ta=16, 200 epochs) is still running on the L20: its 2026-08-18 18:47
CST snapshot is epoch 90/200 with offline sampled L1/L2 0.1135/0.0215. It has no final
checkpoint or rollout yet, so those intermediate numbers are not a policy conclusion. Phase 1
remains incomplete until a rollout success criterion is met or the remaining evidence points to
the next architecture change.

## References

- Diffusion Policy (PushT task, dataset, benchmark): Chi et al., arXiv:2303.04137
- π0 (flow-matching action head): Physical Intelligence, arXiv:2410.24164
- RT-2 (action tokenization): Brohan et al., arXiv:2307.15818
- mini-diffusion (flow objective / distillation reused here): same author
- Dataset: <https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip>

## License

MIT
