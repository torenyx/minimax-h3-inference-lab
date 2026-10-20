"""Bound AdaLN/residual temporaries while retaining resident INT8 PDD8 weights."""
import json
import os
from pathlib import Path
import runpy
import time

import torch
from lightx2v.common.ops.mm import mm_weight
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner

ROWS = 4096
original_convrot = mm_weight.MMWeightWint8ConvRot.apply
lora_checked = False


def convrot_with_bounded_lora(self, input_tensor):
    global lora_checked
    if not self.has_lora_branch:
        return original_convrot(self, input_tensor)
    if input_tensor.shape[-1] % self._convrot_groupsize:
        raise ValueError('INT8 ConvRot input width is not divisible by its group size')
    output = mm_weight._int8_convrot_linear(
        input_tensor.contiguous(), self.weight.contiguous(), self.weight_scale.contiguous(),
        self._get_actual_bias(), self.infer_dtype, self._convrot_groupsize,
    )
    for start in range(0, len(input_tensor), 16384):
        end = min(start + 16384, len(input_tensor))
        output[start:end].add_(self.apply_lora(input_tensor[start:end]))
    if not lora_checked:
        expected = original_convrot(self, input_tensor[:512])
        actual = output[:512]
        relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        assert relative.item() <= 0.01 and torch.isfinite(actual).all()
        print('H3_LORA_ROW_ADD_CHECK ' + json.dumps({'relative_l2': relative.item()}), flush=True)
        lora_checked = True
    return output


def modulate_(output, scale, shift, indices):
    for start in range(0, len(output), ROWS):
        end = min(start + ROWS, len(output))
        select = indices[start:end]
        output[start:end].mul_(1.0 + scale.index_select(0, select))
        output[start:end].add_(shift.index_select(0, select))
    return output


def residual_(output, gate, residual, indices):
    for start in range(0, len(output), ROWS):
        end = min(start + ROWS, len(output))
        output[start:end].mul_(gate.index_select(0, indices[start:end]))
        output[start:end].add_(residual[start:end])
    return output


def self_test():
    # Include a partial last chunk and repeated modality indices.
    generator = torch.Generator().manual_seed(20261016)
    value = torch.randn(ROWS + 7, 16, generator=generator, dtype=torch.bfloat16)
    scale, shift, gate = torch.randn(3, 4, 16, generator=generator, dtype=value.dtype)
    indices = torch.arange(len(value)) % 4
    expected = value * (1.0 + scale[indices]) + shift[indices]
    assert torch.equal(modulate_(value.clone(), scale, shift, indices), expected)
    expected = value + gate[indices] * value
    assert torch.equal(residual_(value.clone(), gate, value, indices), expected)
    print('H3_RESIDENT_POINTWISE_CHECK equal=True', flush=True)


def infer_block(self, weights, hidden_states, pre_infer_out, modulation=None):
    if modulation is None:
        modulation = self._compute_adaln_table(weights, pre_infer_out)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    indices = pre_infer_out.adaln_indices

    normed = modulate_(weights.norm1.apply(hidden_states), scale_msa, shift_msa, indices)
    attention = self._attention(weights.attn, normed, pre_infer_out)
    del normed
    hidden_states = residual_(attention, gate_msa, hidden_states, indices)
    del attention

    normed = modulate_(weights.norm2.apply(hidden_states), scale_mlp, shift_mlp, indices)
    output = self._ff(weights.ff, normed)
    del normed
    return residual_(output, gate_mlp, hidden_states, indices)


original_init = MiniMaxH3Runner.init_run
original_segment = MiniMaxH3Runner.run_segment


def init_run(self):
    self.maybe_empty_cache(force=True, collect_garbage=True)
    original_init(self)
    assert not self.model.block_offload, 'Resident benchmark cannot use block offload'
    assert self.model.transformer_infer.infer_func.__name__ == 'infer_without_offload'
    torch.cuda.synchronize()
    print('H3_RESIDENT_READY ' + json.dumps({
        'rank': int(os.environ['RANK']),
        'block_offload': self.model.block_offload,
        'infer_func': self.model.transformer_infer.infer_func.__name__,
        'allocated_gib': torch.cuda.memory_allocated() / 2**30,
        'reserved_gib': torch.cuda.memory_reserved() / 2**30,
    }), flush=True)


def run_segment(self, segment_idx=0):
    original_step_pre = self.scheduler.step_pre
    def step_pre(index):
        # Release unused scratch blocks between steps; live DiT weights stay resident.
        if index > 0 and int(os.environ['WORLD_SIZE']) == 2:
            torch.cuda.empty_cache()
        return original_step_pre(index)
    self.scheduler.step_pre = step_pre
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = original_segment(self, segment_idx)
    finally:
        self.scheduler.step_pre = original_step_pre
    torch.cuda.synchronize()
    print('H3_RESIDENT_DENOISE ' + json.dumps({
        'rank': int(os.environ['RANK']), 'steps': self.scheduler.infer_steps,
        'seconds': time.perf_counter() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'block_offload': self.model.block_offload,
    }), flush=True)
    return result


self_test()
if int(os.environ.get('WORLD_SIZE', '1')) == 2:
    mm_weight.MMWeightWint8ConvRot.apply = convrot_with_bounded_lora
MiniMaxH3TransformerInfer.infer_block = infer_block
MiniMaxH3Runner.init_run = init_run
MiniMaxH3Runner.run_segment = run_segment
