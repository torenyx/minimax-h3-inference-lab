#!/usr/bin/env python3
"""Create pruned BF16 and INT8 ConvRot checkpoints from official H3 weights."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.accelerator_compat import install
install()
runpy.run_module('runtime.convert_checkpoint', run_name='__main__')
