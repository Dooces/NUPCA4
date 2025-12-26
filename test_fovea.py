import numpy as np
from nupca3.geometry.fovea import init_fovea_state, select_fovea
from nupca3.config import AgentConfig


def test_select_fovea_returns_k_blocks():
    cfg = AgentConfig(D=16, B=4, fovea_blocks_per_step=2)
    f = init_fovea_state(cfg)
    blocks = select_fovea(f, cfg)
    assert len(blocks) == 2
