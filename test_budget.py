from nupca3.control.budget import compute_budget_and_horizon
from nupca3.config import AgentConfig


def test_horizon_zero_in_rest():
    cfg = AgentConfig(B_rt=1.0, b_enc_base=0.1, b_roll_base=0.1)
    bd = compute_budget_and_horizon(rest=True, cfg=cfg, L_eff=0.5)
    assert bd.h == 0
