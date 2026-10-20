"""Normalize modern type annotations for torch 2.5 custom-op schemas only."""
import inspect
import types
import typing


def enable():
    import torch
    # Torch 2.5's CUDA RMSNorm path fails in H3 VAE decode on driver 525.
    # Keep the same normalization with ordinary CUDA reductions instead.
    if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) <= (2, 5):
        from torch.nn import functional as F
        original_rms_norm = F.rms_norm
        if not getattr(original_rms_norm, '_h3_compat', False):
            def rms_norm(value, normalized_shape, weight=None, eps=None):
                if not value.is_cuda:
                    return original_rms_norm(value, normalized_shape, weight, eps)
                dims = tuple(range(value.ndim - len(normalized_shape), value.ndim))
                epsilon = torch.finfo(value.dtype).eps if eps is None else eps
                work = value if value.dtype == torch.float64 else value.float()
                output = work * torch.rsqrt(work.square().mean(dims, keepdim=True) + epsilon)
                if weight is not None:
                    output = output * weight.to(work.dtype)
                return output.to(value.dtype)
            rms_norm._h3_compat = True
            F.rms_norm = rms_norm
    # INT8 ConvRot is supplied by comfy-kitchen, not Transformers TorchAO.
    from transformers.utils import import_utils
    import_utils._torchao_available = False
    from transformers.processing_utils import ProcessorMixin
    if not hasattr(ProcessorMixin, 'create_mm_token_type_ids'):
        def create_mm_token_type_ids(self, input_ids):
            # Same modality IDs as the upstream Transformers v5 processor.
            image = getattr(self, 'image_token_id', None)
            video = getattr(self, 'video_token_id', None)
            audio = getattr(self, 'audio_token_id', None)
            return [[1 if value == image else 2 if value == video else 3 if value == audio else 0 for value in row] for row in input_ids]
        ProcessorMixin.create_mm_token_type_ids = create_mm_token_type_ids
    import torch.distributed.fsdp as fsdp
    if not hasattr(fsdp, 'FSDPModule'):
        from torch.distributed._composable.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
        fsdp.FSDPModule = FSDPModule
        fsdp.fully_shard = fully_shard
        fsdp.MixedPrecisionPolicy = MixedPrecisionPolicy
        fsdp.CPUOffloadPolicy = CPUOffloadPolicy
    original = torch.library.infer_schema
    if getattr(original, '_h3_compat', False):
        return

    def normalize(annotation):
        origin = typing.get_origin(annotation)
        arguments = typing.get_args(annotation)
        aliases = {list: typing.List, tuple: typing.Tuple, dict: typing.Dict}
        if origin in aliases:
            return aliases[origin][tuple(normalize(arg) for arg in arguments)]
        if origin in (types.UnionType, typing.Union):
            return typing.Union[tuple(normalize(arg) for arg in arguments)]
        return annotation

    def infer_schema(function, *args, **kwargs):
        previous = function.__annotations__
        try:
            function.__annotations__ = {name: normalize(value) for name,value in inspect.get_annotations(function,eval_str=True).items()}
            return original(function,*args,**kwargs)
        finally:
            function.__annotations__ = previous

    infer_schema._h3_compat = True
    torch.library.infer_schema = infer_schema
