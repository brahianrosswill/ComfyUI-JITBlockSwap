# ComfyUI-JITBlockSwap

Block Swap (RAM Offload) node for native ComfyUI `MODEL` — run DiT models
larger than VRAM (e.g. Wan 2.2 / Bernini-R 14B fp16, 28.6 GB on a 24 GB GPU)
by keeping the first N transformer blocks in system RAM and streaming each
block to the GPU only while its forward runs.

## Nodes

Two separate nodes so the battle-tested Wan path never shares code with the
newer LTX/fp8 path:

| node | class | targets | code |
|---|---|---|---|
| **Block Swap (RAM Offload)** | `BlockSwap` | Wan 2.2 / Bernini-R (`.blocks`, plain fp16/bf16) | `blockswap.py` — frozen at the Wan-verified implementation |
| **Block Swap LTX (RAM Offload)** | `BlockSwapLTX` | LTX-2.3 22B AV (`.transformer_blocks`, incl. comfy_kitchen fp8 QuantizedTensor) | `blockswap_ltx.py` — adds wrapper-subclass (fp8 QuantizedTensor) swap and a base.to() unpin guard |

Shared inputs:

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
- `BlockSwap` targets `.blocks` (Wan-family, tested); `BlockSwapLTX` also
  detects `.transformer_blocks` (LTX-2.3 22B AV). Flux double blocks are NOT
  covered.
- Do not chain two BlockSwap nodes on the same model.
- Approx. cost: one PCIe H2D transfer per swapped block per forward call
  (~25-30 ms per 660 MB fp16 block on PCIe 4.0 x16).

## LTX-2.3 22B (fp8) support

Tested 2026-07-10 on RTX 4090 24 GB with `ltx-2.3-22b-dev-fp8.safetensors`
(48 `transformer_blocks`): 360×360×49f I2V completed in **40.2 s** with
`blocks_to_swap=12` (auto-raised to 14 to fit the weight budget), 17.9 GB
resident vs a fully packed card without the node (47.4 s).

fp8 checkpoints store weights as `comfy_kitchen` QuantizedTensor wrapper
subclasses, which need special handling implemented here:

- Wrapper subclasses cannot be moved with the `.data` repointing trick (their
  device is fixed at construction and the quantized payload lives in
  `_qdata`); swapped QuantizedTensor params are exchanged at the module
  attribute level instead.
- A sibling ModelPatcher clone (a chain without BlockSwap) calling
  `unpatch_model → model.to()` makes `Module._apply` rebuild the wrappers,
  which would free a still-pinned `_qdata` and poison the CUDA context
  (`CUDA error: invalid argument` much later). The node guards the base
  model's `.to()` to unpin first.

## Known limitation (BlockSwapLTX + 40 GB-class checkpoints)

Very large checkpoints (LTX-2.3 22B bf16, 43 GB, 30/48 blocks swapped) can
fail with `CUDA error: invalid argument` on the next device transfer even
though every ON_LOAD phase synchronizes cleanly. Root cause: **Windows
commit-charge exhaustion**, not a CUDA bug — the swap masters plus LoRA
backups add ~40 GB of committed CPU memory on top of the model, the text
encoder and WDDM's backing reservations, and once the commit limit
(RAM + pagefile) is hit the driver refuses the staging allocation behind the
copy (measured: 161.9 / 166.3 GB committed at failure with 30 GB of physical
RAM still free; a synchronize needs no new commit and passes, the next copy
does not). Smaller flows (fp8 29 GB / Wan 14B fp16) fit and are unaffected.

Workaround: set a large fixed pagefile (e.g. 64 GB — commit is reserved, not
written, so this costs no I/O), or run such checkpoints without BlockSwap.
After a failed run, restart ComfyUI before queueing anything else: the dead
model's commit lingers and can take the next, otherwise-fine job down with
it.

**Caveat — fp8 + runtime LoRA:** with `--disable-dynamic-vram`, ComfyUI's
legacy loader merges LoRA into fp8 weights per-key on a VRAM-packed GPU and
can crawl for hours *before* this node's `ON_LOAD` hook ever runs. BlockSwap
cannot help there. Either run fp8+LoRA workflows with dynamic VRAM (default,
where this node is a no-op), or pre-merge the LoRA (bf16 merge → requantize
to fp8) and drop the LoRA loader from the chain. fp16/bf16 + LoRA are
unaffected.
