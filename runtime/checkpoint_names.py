"""Normalize historical H3 checkpoint names without changing tensor values."""


def normalize_final_norm(weights):
    old, new = 'final_layer.norm.weight', 'norm_out.norm.weight'
    if old in weights and new not in weights:
        weights[new] = weights.pop(old)
    return weights
