# LightX2V patch

Base: ModelTC/LightX2V commit `8e86555809efd20138d3a4e1f937ee306cdf0ee0`.

`lightx2v-h3.patch` contains the two source changes present in the successful
runtime: paired-lane reloads in the fused QKV kernel for Triton compatibility,
and early release of Q/K/V before the attention output projection.

`scripts/setup.sh` applies it to a separate local checkout. Row chunking,
resident AdaLN/residual operations, TP2 checkpoint loading, adapter cache
construction, and decode memory changes are in the parent `runtime/` modules.
