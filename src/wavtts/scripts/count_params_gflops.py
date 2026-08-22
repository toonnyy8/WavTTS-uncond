import os
import sys


sys.path.append(os.getcwd())

import thop
import torch

from wavtts.model import CFM, DiT
from wavtts.model.backbones.dit import STATE_CLEAN


""" ~155M """
# transformer =       DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2)
# transformer =       DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2, long_skip_connection = True)

""" ~335M """
# FLOPs: 622.1 G, Params: 333.2 M
# FLOPs: 363.4 G, Params: 335.8 M
transformer = DiT(dim=1024, depth=22, heads=16, ff_mult=2)


model = CFM(transformer=transformer)  # sanity: constructor still wires up end-to-end
target_sample_rate = 16000
wav_frame_len = 160
duration = 20
num_samples = duration * target_sample_rate

x = torch.randn(1, num_samples)
state = torch.full((1,), STATE_CLEAN, dtype=torch.long)
time = torch.tensor(0.5)

flops, params = thop.profile(transformer, inputs=(x, state, time))
print(f"FLOPs: {flops / 1e9} G")
print(f"Params: {params / 1e6} M")
