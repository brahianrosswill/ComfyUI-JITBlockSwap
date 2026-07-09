# ComfyUI-JITBlockSwap

Block Swap (RAM Offload) node for native ComfyUI `MODEL` — run DiT models
larger than VRAM (e.g. Wan 2.2 / Bernini-R 14B fp16, 28.6 GB on a 24 GB GPU)
by keeping the first N transformer blocks in system RAM and streaming each
block to the GPU only while its forward runs.

## Node

`advanced/model` → **Block Swap (RAM Offload)** (`BlockSwap`)

| input | default | meaning |
|---|---|---|
| `model` | — | MODEL from UNETLoader (place AFTER LoRA loaders) |
| `blocks_to_swap` | 20 | blocks kept in RAM (Wan 14B has 40). Raised automatically if the resident part still exceeds the VRAM weight budget |
| `pin_memory` | true | page-lock CPU copies for fast PCIe transfer |

## How it works

- Hooks `ModelPatcher` `ON_LOAD`: after ComfyUI's normal load pass it
  re-organizes the diffusion model — swap blocks get their weight patches
  (LoRA) baked in once, moved to (pinned) RAM; everything else is made fully
  GPU-resident, removing the slow per-layer LowVramPatch cast path.
- Each swapped block's `forward` is wrapped: params are repointed to a GPU
  copy just-in-time and repointed back to the CPU master afterwards (no
  device-to-host copy needed — weights never change during inference).
  Transfers are synchronized before compute: async submission of large pinned
  H2D copies interleaved with DiT kernels busy-loop hangs the GPU on
  Windows/WDDM (observed: 100% util at ~100 W). Cost is ~0 when compute-bound
  (measured 28-29 s/it either way on Wan 14B).
- `ON_DETACH` restores state; placement is re-applied on every load, so
  model switching (high/low noise) and LoRA changes are safe.

## Notes / limitations

- Chain: `UNETLoader → LoraLoaderModelOnly → BlockSwap → ModelSamplingSD3 → KSampler`.
- **Requires launching ComfyUI with `--disable-dynamic-vram`** (legacy
  ModelPatcher). Without the flag the node does NOT error or stop the
  workflow — it passes the model through unchanged and prints
  `[BlockSwap] dynamic VRAM patcher detected, skipping.` to the console
  only (nothing is shown in the browser UI). The run then relies on
  ComfyUI's standard memory management: small jobs may still succeed,
  large models will OOM or fall back to slow partial loading. If block
  swap "doesn't seem to work", check the console for this line first.
- Works on any model whose diffusion model exposes `.blocks` (Wan, LTX-style
  DiTs, Flux double blocks are NOT covered — Wan-family tested).
- Do not chain two BlockSwap nodes on the same model.
- Approx. cost: one PCIe H2D transfer per swapped block per forward call
  (~25-30 ms per 660 MB fp16 block on PCIe 4.0 x16).
