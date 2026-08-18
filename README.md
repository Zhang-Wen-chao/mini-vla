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

## Quickstart (Phase 0)

```bash
# download + extract the official PushT zarr (~1.5 GB)
bash scripts/download_pusht.sh

# inspect the dataset (shapes, episode count, normalization stats)
python -m mini_vla.data --data-dir data/pusht_cchi_v7_replay.zarr

# CPU unit tests (no GPU, no dataset download needed — synthetic zarr in tmp)
pytest -q
```

## References

- Diffusion Policy (PushT task, dataset, benchmark): Chi et al., arXiv:2303.04137
- π0 (flow-matching action head): Physical Intelligence, arXiv:2410.24164
- RT-2 (action tokenization): Brohan et al., arXiv:2307.15818
- mini-diffusion (flow objective / distillation reused here): same author
- Dataset: <https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip>

## License

MIT
