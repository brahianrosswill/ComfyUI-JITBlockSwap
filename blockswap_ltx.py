"""Block Swap (RAM Offload) for native ComfyUI MODEL.

Keeps the first N transformer blocks of a DiT (Wan 2.2 / Bernini-R,
LTX-2.3 22B AV, etc.) resident in system RAM (optionally pinned) and
streams each block to the GPU only for the duration of its forward call.
Weight patches (LoRA) are baked into the CPU-resident weights once at load
time, so the slow per-layer LowVramPatch cast path is avoided entirely.

The repeated-block ModuleList is looked up under either of two attribute
names (see _BLOCK_ATTR_CANDIDATES below) since different model families
name it differently: Wan 2.2 / Bernini-R use "blocks", LTX-2.3 uses
"transformer_blocks".

Designed against ComfyUI v0.26.x legacy ModelPatcher (--disable-dynamic-vram).
With dynamic VRAM enabled the node is a no-op (dynamic mode manages
placement itself).
"""

import functools
import logging
import os

_BS_DEBUG_SYNC = os.environ.get("BS_DEBUG_SYNC") == "1"


def _dbg_sync(tag):
    if not _BS_DEBUG_SYNC:
        return
    try:
        torch.cuda.synchronize()
        logging.info("[BlockSwapDBG] sync OK: %s", tag)
    except Exception as e:
        logging.error("[BlockSwapDBG] sync FAILED at %s: %s", tag, e)
        raise

import torch

import comfy.model_management as mm
import comfy.model_patcher
from comfy.patcher_extension import CallbacksMP

try:
    from comfy.quant_ops import QuantizedTensor
except Exception:  # older ComfyUI without quant_ops
    QuantizedTensor = ()

CALLBACK_KEY = "blockswap_ltx_ram_offload"

# transient VRAM headroom for the block(s) in flight during swapping
_SWAP_BUFFER_EXTRA = 128 * 1024 * 1024

# Name of the ModuleList holding the repeated transformer blocks. Differs by
# model family: Wan 2.2 / Bernini-R use "blocks", LTX-2.3 (diffusers-style
# naming, see comfy/ldm/lightricks/model.py _init_transformer_blocks) uses
# "transformer_blocks". Tried in order, first match wins.
_BLOCK_ATTR_CANDIDATES = ("blocks", "transformer_blocks")


def _find_blocks(dm):
    for name in _BLOCK_ATTR_CANDIDATES:
        blocks = getattr(dm, name, None)
        if blocks is not None:
            return name, blocks
    return None, None


def _make_swap_forward(block):
    orig_forward = block._bs_orig_forward

    def swap_forward(*args, **kwargs):
        state = getattr(block, "_bs_state", None)
        if state is None or not state.get("active", False):
            return orig_forward(*args, **kwargs)
        load_device = state["load_device"]
        masters = state["masters"]
        non_blocking = not state.get("sync_transfers", True)
        for entry in masters:
            if entry[0] == "data":
                _, t, master = entry
                t.data = master.to(load_device, non_blocking=non_blocking)
            else:
                # wrapper subclasses (comfy_kitchen QuantizedTensor): device is
                # baked into the wrapper at construction, so .data swap cannot
                # move them - swap the module attribute with a GPU copy instead
                _, module, name, master, is_param = entry
                moved = master.to(load_device, non_blocking=non_blocking)
                if is_param:
                    module._parameters[name] = torch.nn.Parameter(moved, requires_grad=False)
                else:
                    module._buffers[name] = moved
        if not non_blocking:
            # serialize the queue before compute: async submission of large
            # pinned H2D copies interleaved with DiT kernels busy-loop hangs
            # the GPU on this Windows/WDDM setup (100% util at idle power)
            torch.cuda.synchronize()
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for entry in masters:
                if entry[0] == "data":
                    _, t, master = entry
                    t.data = master
                else:
                    _, module, name, master, is_param = entry
                    if is_param:
                        module._parameters[name] = master
                    else:
                        module._buffers[name] = master

    return swap_forward


def _collect_swap_entries(block):
    """Enumerate the block's tensors as swap entries.

    Plain tensors use the in-place .data swap (no attribute churn, D2H-free
    restore). Wrapper subclasses (QuantizedTensor) must be swapped at the
    module-attribute level, and their pinnable CPU storage is the inner
    _qdata rather than the wrapper."""
    entries = []
    for mod_name, m in block.named_modules():
        for name, p in list(m.named_parameters(recurse=False)):
            if isinstance(p, QuantizedTensor) or isinstance(getattr(p, "data", None), QuantizedTensor):
                master = m._parameters[name]
                entries.append(("attr", m, name, master, True))
            else:
                entries.append(("data", p, p.data))
        for name, b in list(m.named_buffers(recurse=False)):
            if isinstance(b, QuantizedTensor):
                entries.append(("attr", m, name, b, False))
            else:
                entries.append(("data", b, b.data))
    return entries


def _entry_pin_targets(entry):
    """CPU tensors of this entry that can be page-locked."""
    if entry[0] == "data":
        return [entry[2]]
    master = entry[3]
    inner = getattr(master, "_qdata", None)
    return [inner] if inner is not None else []


def _finalize_module(patcher, name, module, target_device, unpin_all=False):
    """Bake pending weight patches for `module` (leaf params only) onto
    target_device and remove the lowvram cast path. Idempotent.

    unpin_all must be True whenever the module is about to move off the CPU:
    comfy's load() pins offloaded weights (cudaHostRegister) and a pinned
    tensor that gets freed while still registered leaves a dangling
    registration that later surfaces as CUDA error: invalid argument."""
    params = dict(module.named_parameters(recurse=False))
    for pname in params:
        key = "{}.{}".format(name, pname) if name else pname
        if unpin_all or key in patcher.patches:
            patcher.unpin_weight(key)
    if getattr(module, "comfy_patched_weights", False) is not True:
        for pname in params:
            key = "{}.{}".format(name, pname) if name else pname
            if key in patcher.patches:
                patcher.patch_weight_to_device(key, device_to=target_device)
        if params or hasattr(module, "comfy_cast_weights"):
            module.comfy_patched_weights = True
    # weights are baked; per-forward cast/patch functions must not run again
    comfy.model_patcher.wipe_lowvram_weight(module)


def _finalize_tree(patcher, prefix, root, target_device, unpin_all=False):
    for sub_name, m in root.named_modules():
        full = "{}.{}".format(prefix, sub_name) if sub_name else prefix
        _finalize_module(patcher, full, m, target_device, unpin_all=unpin_all)
    root.to(target_device)


def _deactivate_block(block, unpin=True):
    state = getattr(block, "_bs_state", None)
    if state is None:
        return
    state["active"] = False
    if unpin:
        for t in state.get("pinned", []):
            mm.unpin_memory(t)
        state["pinned"] = []
    block._bs_state = None


def _guard_base_to(base, dm):
    """Ensure our pins never outlive a weight move we don't control.

    A sibling ModelPatcher clone (e.g. a LoRA chain without BlockSwap) calls
    unpatch_model -> base.to(offload) on the SHARED torch model. Module._apply
    rebuilds wrapper-subclass params (QuantizedTensor) even for a same-device
    move, freeing the inner _qdata while it is still cudaHostRegister'ed - the
    dangling registration then kills an unrelated CUDA call with
    'invalid argument'. Deactivate (and unpin) all swap state before any
    whole-model .to()."""
    if getattr(base, "_bs_to_guarded", False):
        return
    orig_to = base.to

    def guarded_to(*args, **kwargs):
        _, blocks = _find_blocks(dm)
        if blocks is not None:
            for block in blocks:
                _deactivate_block(block)
        return orig_to(*args, **kwargs)

    base.to = guarded_to
    base._bs_to_guarded = True


def _on_load(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load,
             blocks_to_swap=0, pin_masters=True):
    if patcher.is_dynamic():
        logging.warning("[BlockSwap] dynamic VRAM patcher detected, skipping. "
                        "Run ComfyUI with --disable-dynamic-vram to use block swap.")
        return
    base = patcher.model
    dm = getattr(base, "diffusion_model", None)
    block_attr, blocks = _find_blocks(dm)
    if blocks is None:
        logging.warning("[BlockSwap] model has no diffusion_model.blocks / "
                        ".transformer_blocks, skipping.")
        return

    offload_device = patcher.offload_device
    load_device = torch.device(device_to if device_to is not None else patcher.load_device)
    if not mm.is_device_cuda(load_device):
        logging.warning("[BlockSwap] load device is not CUDA, skipping.")
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
            logging.info("[BlockSwap] blocks_to_swap raised {} -> {} to fit the "
                         "{:.0f} MB VRAM weight budget.".format(
                             blocks_to_swap, n_swap, lowvram_model_memory / (1024 * 1024)))

    prefix = "diffusion_model.{}".format(block_attr)

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
        if not hasattr(block, "_bs_orig_forward"):
            block._bs_orig_forward = block.forward
            block.forward = _make_swap_forward(block)
        block._bs_state = {
            "active": True,
            "load_device": load_device,
            "masters": masters,
            "pinned": pinned,
        }
        swapped_bytes += block_sizes[i]
        _dbg_sync("swap block {}".format(i))

    # 2) make everything else fully GPU-resident (no per-layer cast path)
    for i in range(n_swap, total):
        block = blocks[i]
        _deactivate_block(block)
        _finalize_tree(patcher, "{}.{}".format(prefix, i), block, load_device, unpin_all=True)
        _dbg_sync("resident block {}".format(i))

    for sub_name, m in dm.named_modules():
        if sub_name == block_attr or sub_name.startswith(block_attr + "."):
            continue
        full = "diffusion_model.{}".format(sub_name) if sub_name else "diffusion_model"
        _finalize_module(patcher, full, m, load_device, unpin_all=True)
        if sub_name and "." not in sub_name:
            m.to(load_device)
            _dbg_sync("non-block module {}".format(sub_name))
    # root-level tensors of the diffusion model itself
    for t in list(dm.parameters(recurse=False)) + list(dm.buffers(recurse=False)):
        t.data = t.data.to(load_device)

    resident = dm_size - swapped_bytes
    base.model_lowvram = n_swap > 0
    base.model_loaded_weight_memory = resident
    base.model_offload_buffer_memory = (max(block_sizes) if n_swap > 0 else 0) + _SWAP_BUFFER_EXTRA

    _dbg_sync("root tensors")
    mm.soft_empty_cache()
    _dbg_sync("after soft_empty_cache")
    logging.info("[BlockSwap] {} / {} blocks swapped to RAM: {:.0f} MB offloaded"
                 " ({:.0f} MB pinned), {:.0f} MB resident on GPU.".format(
                     n_swap, total, swapped_bytes / (1024 * 1024),
                     pinned_bytes / (1024 * 1024), resident / (1024 * 1024)))


def _on_detach(patcher, unpatch_all):
    dm = getattr(patcher.model, "diffusion_model", None)
    _, blocks = _find_blocks(dm)
    if blocks is None:
        return
    for block in blocks:
        _deactivate_block(block)


class BlockSwapLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": ("INT", {
                    "default": 20, "min": 0, "max": 80, "step": 1,
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
    DESCRIPTION = ("Streams the first N transformer blocks from system RAM to the GPU "
                   "on demand so models larger than VRAM can run. LoRA patches are "
                   "baked in once at load time.")

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
    "BlockSwapLTX": BlockSwapLTX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlockSwapLTX": "Block Swap LTX (RAM Offload)",
}
