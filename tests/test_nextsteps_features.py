import numpy as np

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.geometry.fovea import select_fovea
from nupca3.memory.salience import compute_scores
from nupca3.step_pipeline import (
    _compute_peripheral_gist,
    _update_block_signals,
    _update_context_register,
    _update_coverage_debts,
)
from nupca3.types import EnvObs, FoveaState


def test_coverage_expert_debt_biases_scores() -> None:
    cfg = AgentConfig(
        D=8,
        B=2,
        fovea_blocks_per_step=1,
        alpha_pi=0.0,
        alpha_deg=0.0,
        alpha_ctx_relevance=0.0,
        alpha_ctx_gist=0.0,
        alpha_cov_exp=1.0,
        alpha_cov_band=0.0,
        beta_context=0.0,
        beta_context_node=0.0,
    )
    agent = NUPCA3Agent(cfg)
    state = agent.state
    state.active_set = {0}

    _update_coverage_debts(state, cfg)
    # pick a non-zero node to compare coverage effect
    other = next(nid for nid in state.coverage_expert_debt if nid != 0)

    state.coverage_expert_debt[0] = 5
    state.coverage_expert_debt[other] = 0

    scores = compute_scores(state, cfg, observed_dims=set())
    assert scores[0] > scores[other], "Expert debt should boost node score"


def test_coverage_band_debt_biases_scores() -> None:
    cfg = AgentConfig(
        D=8,
        B=2,
        fovea_blocks_per_step=1,
        alpha_pi=0.0,
        alpha_deg=0.0,
        alpha_ctx_relevance=0.0,
        alpha_ctx_gist=0.0,
        alpha_cov_exp=0.0,
        alpha_cov_band=1.0,
        beta_context=0.0,
        beta_context_node=0.0,
    )
    agent = NUPCA3Agent(cfg)
    state = agent.state
    state.active_set = {0}

    _update_coverage_debts(state, cfg)
    level_to_nodes: dict[int, list[int]] = {}
    for nid, level in state.node_band_levels.items():
        level_to_nodes.setdefault(level, []).append(nid)

    assert len(level_to_nodes) >= 2, "Need at least two abstraction levels for this test"

    low_level, high_level = sorted(level_to_nodes)[:2]
    low_node = level_to_nodes[low_level][0]
    high_node = level_to_nodes[high_level][0]

    state.coverage_expert_debt[low_node] = 0
    state.coverage_expert_debt[high_node] = 0
    state.coverage_band_debt[low_level] = 0
    state.coverage_band_debt[high_level] = 5

    scores = compute_scores(state, cfg, observed_dims=set())
    assert scores[high_node] > scores[low_node], "Band debt should bias higher-level nodes"


def test_peripheral_gist_updates_context_register() -> None:
    cfg = AgentConfig(
        D=20,
        B=4,
        periph_blocks=1,
        periph_bins=2,
        beta_context=0.5,
    )
    agent = NUPCA3Agent(cfg)
    state = agent.state

    x_prev = np.arange(cfg.D, dtype=float)
    gist = _compute_peripheral_gist(x_prev, cfg)

    assert gist.size > 0
    state.context_register = np.zeros_like(gist)
    _update_context_register(state, gist, cfg)

    expected = gist * cfg.beta_context
    assert np.allclose(state.context_register, expected)


def test_uncertainty_drives_fovea_selection() -> None:
    cfg = AgentConfig(
        D=4,
        B=4,
        fovea_blocks_per_step=1,
        alpha_cov=0.0,
        fovea_uncertainty_weight=1.0,
    )
    residuals = np.array([0.1, 0.05, 0.0, 0.0], dtype=float)
    uncertainties = np.array([0.0, 10.0, 0.0, 0.0], dtype=float)
    block_age = np.zeros(4, dtype=float)
    fovea = FoveaState(
        block_residual=residuals.copy(),
        block_age=block_age.copy(),
        block_uncertainty=uncertainties.copy(),
        block_costs=np.ones(4, dtype=float),
        routing_scores=np.zeros(4, dtype=float),
    )

    selected_with_uncertainty = select_fovea(fovea, cfg)
    assert selected_with_uncertainty == [1]

    cfg_no_uncertainty = cfg.replace(fovea_uncertainty_weight=0.0)
    selected_without_uncertainty = select_fovea(fovea, cfg_no_uncertainty)
    assert selected_without_uncertainty == [0]


def test_support_window_history_respects_window_size() -> None:
    cfg = AgentConfig(
        D=8,
        B=4,
        fovea_blocks_per_step=1,
        multi_world_support_window=2,
    )
    agent = NUPCA3Agent(cfg)
    for _ in range(5):
        obs = EnvObs(x_partial={0: 0.1})
        agent.step(obs)
    assert len(agent.state.observed_history) <= 2


def test_peripheral_coherence_residual_with_full_observation() -> None:
    cfg = AgentConfig(
        D=8,
        B=4,
        periph_blocks=1,
        periph_bins=1,
        periph_channels=1,
    )
    agent = NUPCA3Agent(cfg)
    full_obs = EnvObs(
        x_partial={i: float(i + 1) for i in range(cfg.D)},
        x_full=np.arange(cfg.D, dtype=float),
    )
    agent.step(full_obs)
    assert 0.0 <= agent.state.peripheral_confidence <= 1.0
    residual = agent.state.peripheral_residual
    assert np.isfinite(residual)
    assert agent.state.peripheral_prior.size > 0
    assert agent.state.peripheral_obs.size > 0


def test_block_signals_reset_without_worlds() -> None:
    cfg = AgentConfig(D=4, B=2)
    agent = NUPCA3Agent(cfg)
    agent.state.peripheral_residual = 0.0
    _update_block_signals(agent.state, cfg, [], cfg.D)
    assert np.allclose(agent.state.fovea.block_disagreement, 0.0)
    assert np.allclose(agent.state.fovea.block_innovation, 0.0)
    assert np.allclose(agent.state.fovea.block_periph_demand, 0.0)
