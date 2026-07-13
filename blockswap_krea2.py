"""Block Swap (RAM Offload) for Krea 2 (SingleStreamDiT) native ComfyUI MODEL.

Same RAM-offload strategy as blockswap.py / blockswap_ltx.py (first N blocks
live in system RAM, streamed to the GPU per forward, LoRA patches baked into
the CPU masters at load time), but the swap is triggered by module hooks
instead of a ``block.forward`` wrapper.

Why: ComfyUI-Krea2-Ostris-Edit replaces ``diffusion_model.forward`` via
``add_object_patch`` and drives each block's submodules directly
(``block.mod`` / ``block.attn`` / ``block.mlp`` — see ``_block_ref_forward``
and ``_block_kv_forward`` in that repo), so a wrapper on ``block.forward``
never runs on the reference-latent paths and the swapped weights would stay
on the CPU. Every Krea2 block path — native ``SingleStreamBlock.forward``,
the Ostris per-span ref forward, and the Ostris kv_cache forward — starts by
calling ``block.mod`` and ends with ``block.mlp``, so:

  - ``forward_pre_hook`` on ``block.mod``  -> stream the block's weights in
  - ``forward_hook``     on ``block.mlp``  -> point the weights back at the
                                              CPU masters

Designed against ComfyUI legacy ModelPatcher (--disable-dynamic-vram).
With dynamic VRAM enabled the node is a no-op (dynamic mode manages
placement itself).
"""

import functools
import logging

import torch

import comfy.model_management as mm
from comfy.patcher_extension import CallbacksMP

from .blockswap_ltx import (
    _SWAP_BUFFER_EXTRA,
    _collect_swap_entries,
    _entry_pin_targets,
    _finalize_module,
    _finalize_tree,
)

CALLBACK_KEY = "blockswap_krea2_ram_offload"


def _load_block(block):
    state = getattr(block, "_bs_state", None)
    if state is None or not state.get("active", False) or state.get("gpu_loaded", False):
        return
    load_device = state["load_device"]
    for entry in state["masters"]:
        if entry[0] == "data":
            _, t, master = entry
            t.data = master.to(load_device, non_blocking=False)
        else:
            # wrapper subclasses (comfy_kitchen QuantizedTensor): device is
            # baked into the wrapper at construction, so .data swap cannot
            # move them - swap the module attribute with a GPU copy instead
            _, module, name, master, is_param = entry
            moved = master.to(load_device, non_blocking=False)
            if is_param:
                module._parameters[name] = torch.nn.Parameter(moved, requires_grad=False)
            else:
                module._buffers[name] = moved
    # serialize the queue before compute: async submission of large pinned
    # H2D copies interleaved with DiT kernels busy-loop hangs the GPU on
    # Windows/WDDM (see blockswap.py)
    torch.cuda.synchronize()
    state["gpu_loaded"] = True


def _release_block(block):
    state = getattr(block, "_bs_state", None)
    if state is None or not state.get("gpu_loaded", False):
        return
    for entry in state["masters"]:
        if entry[0] == "data":
            _, t, master = entry
            t.data = master
        else:
            _, module, name, master, is_param = entry
            if is_param:
                module._parameters[name] = master
            else:
                module._buffers[name] = master
    state["gpu_loaded"] = False


def _install_hooks(block):
    if getattr(block, "_bs_hooks", None):
        return

    def _pre(module, args):
        _load_block(block)

    def _post(module, args, output):
        _release_block(block)

    block._bs_hooks = [
        block.mod.register_forward_pre_hook(_pre),
        block.mlp.register_forward_hook(_post),
    ]


def _deactivate_block(block, unpin=True):
    state = getattr(block, "_bs_state", None)
    if state is None:
        return
    # restore the CPU masters first: an interrupted run can leave the block
    # pointing at its transient GPU copies
    _release_block(block)
    state["active"] = False
    if unpin:
        for t in state.get("pinned", []):
            mm.unpin_memory(t)
        state["pinned"] = []
    block._bs_state = None


def _guard_base_to(base, dm):
    """Ensure our pins never outlive a weight move we don't control.

    A sibling ModelPatcher clone (e.g. a LoRA chain without BlockSwap) calls
    unpatch_model -> base.to(offload) on the SHARED torch model, which can
    free pinned storages while still cudaHostRegister'ed (see
    blockswap_ltx.py). Deactivate all swap state before any whole-model
    .to()."""
    if getattr(base, "_bs_k2_to_guarded", False):
        return
    orig_to = base.to

    def guarded_to(*args, **kwargs):
        blocks = getattr(dm, "blocks", None)
        if blocks is not None:
            for block in blocks:
                _deactivate_block(block)
        return orig_to(*args, **kwargs)

    base.to = guarded_to
    base._bs_k2_to_guarded = True


def _on_load(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load,
             blocks_to_swap=0, pin_masters=True):
    if patcher.is_dynamic():
        logging.warning("[BlockSwapKrea2] dynamic VRAM patcher detected, skipping. "
                        "Run ComfyUI with --disable-dynamic-vram to use block swap.")
        return
    base = patcher.model
    dm = getattr(base, "diffusion_model", None)
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        logging.warning("[BlockSwapKrea2] model has no diffusion_model.blocks, skipping.")
        return
    if len(blocks) > 0 and not (hasattr(blocks[0], "mod") and hasattr(blocks[0], "mlp")):
        logging.warning("[BlockSwapKrea2] blocks lack mod/mlp hook anchors (not a "
                        "Krea2 SingleStreamDiT?), skipping.")
        return

    offload_device = patcher.offload_device
    load_device = torch.device(device_to if device_to is not None else patcher.load_device)
    if not mm.is_device_cuda(load_device):
        logging.warning("[BlockSwapKrea2] load device is not CUDA, skipping.")
        return

    _guard_base_to(base, dm)

    total = len(blocks)
    n_swap = max(0, min(int(blocks_to_swap), total))

    block_sizes = [mm.module_size(b) for b in blocks]
    dm_size = mm.module_size(dm)
    other_size = dm_size - sum(block_sizes)

    # respect the VRAM budget computed by comfy: swap more blocks if the
    # requested resident set would not fit
    if lowvram_model_memory is not None and lowvram_model_memory < 1e30:
        margin = max(block_sizes) if block_sizes else 0
        margin += _SWAP_BUFFER_EXTRA

        def resident_bytes(n):
            return other_size + sum(block_sizes[n:])

        while n_swap < total and resident_bytes(n_swap) + margin > lowvram_model_memory:
            n_swap += 1
        if n_swap > blocks_to_swap:
            logging.info("[BlockSwapKrea2] blocks_to_swap raised {} -> {} to fit the "
                         "{:.0f} MB VRAM weight budget.".format(
                             blocks_to_swap, n_swap, lowvram_model_memory / (1024 * 1024)))

    prefix = "diffusion_model.blocks"

    # 1) offload swap blocks first to free VRAM
    swapped_bytes = 0
    pinned_bytes = 0
    for i in range(n_swap):
        block = blocks[i]
        _deactivate_block(block)
        _finalize_tree(patcher, "{}.{}".format(prefix, i), block, offload_device)
        masters = _collect_swap_entries(block)
        pinned = []
        if pin_masters:
            for entry in masters:
                for t in _entry_pin_targets(entry):
                    if mm.pin_memory(t):
                        pinned.append(t)
                        pinned_bytes += t.nbytes
        _install_hooks(block)
        block._bs_state = {
            "active": True,
            "gpu_loaded": False,
            "load_device": load_device,
            "masters": masters,
            "pinned": pinned,
        }
        swapped_bytes += block_sizes[i]

    # 2) make everything else fully GPU-resident (no per-layer cast path)
    for i in range(n_swap, total):
        block = blocks[i]
        _deactivate_block(block)
        _finalize_tree(patcher, "{}.{}".format(prefix, i), block, load_device, unpin_all=True)

    for sub_name, m in dm.named_modules():
        if sub_name == "blocks" or sub_name.startswith("blocks."):
            continue
        full = "diffusion_model.{}".format(sub_name) if sub_name else "diffusion_model"
        _finalize_module(patcher, full, m, load_device, unpin_all=True)
        if sub_name and "." not in sub_name:
            m.to(load_device)
    # root-level tensors of the diffusion model itself
    for t in list(dm.parameters(recurse=False)) + list(dm.buffers(recurse=False)):
        t.data = t.data.to(load_device)

    resident = dm_size - swapped_bytes
    base.model_lowvram = n_swap > 0
    base.model_loaded_weight_memory = resident
    base.model_offload_buffer_memory = (max(block_sizes) if n_swap > 0 else 0) + _SWAP_BUFFER_EXTRA

    mm.soft_empty_cache()
    logging.info("[BlockSwapKrea2] {} / {} blocks swapped to RAM: {:.0f} MB offloaded"
                 " ({:.0f} MB pinned), {:.0f} MB resident on GPU.".format(
                     n_swap, total, swapped_bytes / (1024 * 1024),
                     pinned_bytes / (1024 * 1024), resident / (1024 * 1024)))


def _on_detach(patcher, unpatch_all):
    dm = getattr(patcher.model, "diffusion_model", None)
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        return
    for block in blocks:
        _deactivate_block(block)


class BlockSwapKrea2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": ("INT", {
                    "default": 14, "min": 0, "max": 80, "step": 1,
                    "tooltip": "Number of transformer blocks kept in system RAM and "
                               "streamed to the GPU per forward pass. Raise if you "
                               "still hit OOM. Automatically raised further when the "
                               "resident part would not fit in VRAM."}),
                "pin_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Page-lock the CPU copies for faster PCIe transfers. "
                               "Costs the same amount of non-swappable system RAM."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"
    DESCRIPTION = ("Streams the first N Krea 2 transformer blocks from system RAM to "
                   "the GPU on demand so models larger than VRAM can run. Hook-based, "
                   "so it also works with Krea2OstrisEditModelPatch reference latents. "
                   "LoRA patches are baked in once at load time.")

    def apply(self, model, blocks_to_swap, pin_memory):
        if blocks_to_swap <= 0:
            return (model,)
        model = model.clone()
        model.add_callback_with_key(
            CallbacksMP.ON_LOAD, CALLBACK_KEY,
            functools.partial(_on_load, blocks_to_swap=blocks_to_swap, pin_masters=pin_memory))
        model.add_callback_with_key(CallbacksMP.ON_DETACH, CALLBACK_KEY, _on_detach)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "BlockSwapKrea2": BlockSwapKrea2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlockSwapKrea2": "Block Swap Krea2 (RAM Offload)",
}
