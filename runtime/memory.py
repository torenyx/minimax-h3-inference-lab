"""Bound MiniMax-H3 FFN activations by processing independent token rows."""
import os
import json
import runpy
from pathlib import Path

import torch
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer

chunk_size = int(os.environ['H3_FF_CHUNK_TOKENS'])
assert chunk_size > 0
original_ff = MiniMaxH3TransformerInfer._ff
validated = False
rms_chunk_rows = int(os.environ.get('H3_RMS_CHUNK_ROWS', '0'))
if rms_chunk_rows:
    original_rms = torch.nn.functional.rms_norm
    rms_validated = False

    def chunked_rms(value, normalized_shape, weight=None, eps=None):
        global rms_validated
        if not value.is_cuda or value.ndim <= len(normalized_shape) or value.shape[0] <= rms_chunk_rows:
            return original_rms(value, normalized_shape, weight, eps)
        if not rms_validated:
            sample = value[:1024]
            expected = original_rms(sample, normalized_shape, weight, eps)
            actual = torch.cat([original_rms(rows, normalized_shape, weight, eps) for rows in sample.split(512)])
            equal = torch.equal(expected, actual)
            print(f'H3_RMS_CHUNK_CHECK rank={os.environ.get("RANK", "0")} equal={equal}', flush=True)
            if not equal:
                raise RuntimeError('RMSNorm row chunking changed the validation sample')
            del expected, actual, sample
            rms_validated = True
        output = torch.empty_like(value)
        for start in range(0, len(value), rms_chunk_rows):
            end = min(start + rms_chunk_rows, len(value))
            output[start:end].copy_(original_rms(value[start:end], normalized_shape, weight, eps))
        return output

    torch.nn.functional.rms_norm = chunked_rms


def comparison(expected, actual):
    reference, candidate = expected.float().flatten(), actual.float().flatten()
    difference = candidate - reference
    reference_norm = reference.norm().clamp_min(1e-12)
    return {
        'equal': torch.equal(expected, actual),
        'finite': bool(torch.isfinite(candidate).all().item()),
        'max_abs': difference.abs().max().item(),
        'reference_max_abs': reference.abs().max().item(),
        'relative_l2': (difference.norm() / reference_norm).item(),
        'cosine': torch.nn.functional.cosine_similarity(reference, candidate, dim=0).item(),
    }


def chunked_ff(weights, hidden_states):
    global validated
    if not validated:
        sample = hidden_states[:2 * chunk_size]
        expected = original_ff(weights, sample)
        actual = torch.cat([original_ff(weights, rows) for rows in sample.split(chunk_size)], dim=0)
        metrics = comparison(expected, actual)
        # Experimental numerical gate; passing does not establish end-to-end video equivalence.
        passed = metrics['finite'] and metrics['relative_l2'] <= 0.01 and metrics['cosine'] >= 0.9999
        print('H3_FF_CHUNK_CHECK ' + json.dumps({
            'rank': os.environ.get('RANK', '0'), 'rows': len(sample),
            'chunk_tokens': chunk_size, 'relative_l2_limit': 0.01,
            'cosine_minimum': 0.9999, 'passed': passed, **metrics,
        }), flush=True)
        if not passed:
            raise RuntimeError('FFN chunking exceeded the experimental numerical tolerance')
        del expected, actual, sample
        validated = True
    if hidden_states.shape[0] <= chunk_size:
        return original_ff(weights, hidden_states)
    output = torch.empty_like(hidden_states)
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        output[start:end].copy_(original_ff(weights, hidden_states[start:end]))
    return output


MiniMaxH3TransformerInfer._ff = staticmethod(chunked_ff)
print(f'H3_FF_CHUNK_TOKENS={chunk_size}', flush=True)
