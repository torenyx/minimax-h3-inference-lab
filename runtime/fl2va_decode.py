import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.networks.minimax_h3.packing import unpatchify_video_tokens, unpack_audio_tokens
from lightx2v.models.video_encoders.hf.minimax_h3 import MiniMaxH3VideoVAE
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model

out = Path(os.environ["H3_OUTPUT_DIR"])

# The author's checkpoint exports B@A with no extra alpha/rank attenuation.
original_load_lora = MiniMaxH3Model._load_lora_file


def load_unit_scale_lora(self, file_path, alpha=None):
    weights = original_load_lora(self, file_path, alpha)
    count = 0
    for name, down in weights.items():
        if name.endswith('.lora_down.weight'):
            alpha_name = name.removesuffix('.lora_down.weight') + '.alpha'
            assert float(weights[alpha_name]) == down.shape[0], f'Turbo LoRA scaled incorrectly: {name}'
            count += 1
    assert count == 312, f'Turbo backbone adapter count changed: {count}'
    print(f'H3_TURBO_LORA_SCALE_CHECK pairs={count} scale=1.0 rank={dist.get_rank()}', flush=True)
    return weights


MiniMaxH3Model._load_lora_file = load_unit_scale_lora

original_decode = MiniMaxH3Runner.run_vae_decoder


def decode_after_dit_release(self, video_rows, audio_rows):
    # Persist the completed denoise result so decode retries never repeat DiT.
    if dist.get_rank() == 0:
        scheduler = self.scheduler
        video = unpatchify_video_tokens(
            video_rows[scheduler.num_condition_video_rows:], scheduler.num_latent_frames,
            scheduler.latent_height, scheduler.latent_width,
            channels=int(self.config.get('in_channels', 24)),
            patch_size=tuple(self.config.get('patch_size', (1, 2, 2))),
        )
        audio = unpack_audio_tokens(audio_rows[scheduler.num_condition_audio_rows:], scheduler.num_audio_latents)
        torch.save({'video_latent': video.cpu(), 'audio_latent': audio.cpu()}, out / 'decode-input.pt')
        del video, audio
        print('H3_LIGHT_DENOISE_LATENTS_SAVED', flush=True)
    self.model.to_cpu()
    self.maybe_empty_cache(force=True, collect_garbage=True)
    print(f'H3_LIGHT_DIT_RELEASED_BEFORE_DECODE rank={dist.get_rank()} allocated={torch.cuda.memory_allocated()}', flush=True)
    return original_decode(self, video_rows, audio_rows)


def inplace_video_postprocess(self, video):
    # Preserve multiply/add rounding while avoiding two full-video temporaries.
    mean = self.pixel_mean.to(device=video.device).view(1, -1, 1, 1, 1)
    std = self.pixel_std.to(device=video.device).view(1, -1, 1, 1, 1)
    return video.to(self.sensitive_layer_dtype).mul_(std).add_(mean).clamp_(0, 1)


MiniMaxH3Runner.run_vae_decoder = decode_after_dit_release
MiniMaxH3VideoVAE.postprocess = inplace_video_postprocess
