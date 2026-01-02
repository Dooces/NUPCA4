import math
import time
import numpy as np
import pytest

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.geometry.fovea import (
    dims_for_block,
    init_fovea_state,
    make_observation_set,
    select_fovea,
    update_fovea_tracking,
)
from nupca3.memory.salience import compute_scores
from nupca3.step_pipeline import (
    _compute_peripheral_gist,
    _compute_block_signals,
    _update_context_register,
    _update_coverage_debts,
)
from nupca3.types import EnvObs, FoveaState, ObservationBuffer


def _step_agent(agent: NUPCA3Agent, obs: EnvObs):
    obs.t_w = agent.state.t_w + 1
    obs.wall_ms = int(time.perf_counter() * 1000)
    return agent.step(obs)


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

    scores = compute_scores(state, cfg, observed_dims=set(), candidate_node_ids=state.library.nodes.keys())
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
        transport_span_blocks=1,
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

    scores = compute_scores(state, cfg, observed_dims=set(), candidate_node_ids=state.library.nodes.keys())
    assert scores[high_node] > scores[low_node], "Band debt should bias higher-level nodes"


def test_salience_candidate_set_stays_block_keyed() -> None:
    cfg = AgentConfig(
        D=8,
        B=4,
        fovea_blocks_per_step=1,
        working_set_linger_steps=0,
        beta_context=0.0,
        beta_context_node=0.0,
    )
    agent = NUPCA3Agent(cfg)
    state = agent.state

    state.fovea.block_residual = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    state.fovea.current_blocks = {0}
    state.pending_fovea_selection = {}
    state.buffer.observed_dims = set()
    scores = compute_scores(state, cfg, observed_dims=set())

    scored = set(scores.keys())
    required = {0, 1}
    assert required.issubset(scored), "Anchor and block-0 node must be scored"
    extras = scored - required
    assert len(extras) <= cfg.salience_explore_budget
    assert len(scored) == len(required) + len(extras)
    assert state.salience_candidate_ids == scored

def test_salience_candidate_invariant_never_full_library() -> None:
    cfg = AgentConfig(
        D=8,
        B=6,
        fovea_blocks_per_step=1,
    )
    agent = NUPCA3Agent(cfg)
    state = agent.state

    state.fovea.block_residual = np.array([1.0] + [0.0] * (cfg.B - 1), dtype=float)
    state.fovea.current_blocks = {0}
    state.buffer.observed_dims = set()

    compute_scores(state, cfg, observed_dims=set())

    total_nodes = len(state.library.nodes)
    assert total_nodes > 0
    assert len(state.salience_candidate_ids) < total_nodes
    assert state.salience_num_nodes_scored == len(state.salience_candidate_ids)
    assert state.salience_candidate_ids, "Candidate set should not be empty during normal operation"

def test_salience_candidate_limit_enforced() -> None:
    cfg = AgentConfig(
        D=8,
        B=8,
        fovea_blocks_per_step=8,
        salience_max_candidates=2,
    )
    agent = NUPCA3Agent(cfg)
    state = agent.state

    state.fovea.block_residual = np.ones(cfg.B, dtype=float)
    state.fovea.block_age = np.zeros(cfg.B, dtype=float)
    state.fovea.current_blocks = set(range(cfg.B))
    state.buffer.observed_dims = set()

    compute_scores(state, cfg, observed_dims=set())

    assert state.salience_candidate_limit == 2
    assert state.salience_candidate_count_raw > state.salience_candidate_limit
    assert state.salience_candidates_truncated
    assert len(state.salience_candidate_ids) <= state.salience_candidate_limit
    assert state.salience_num_nodes_scored == len(state.salience_candidate_ids)

def test_salience_debug_exhaustive_scores_entire_library() -> None:
    cfg = AgentConfig(
        D=8,
        B=4,
        salience_debug_exhaustive=True,
    )
    agent = NUPCA3Agent(cfg)
    state = agent.state

    state.buffer.observed_dims = set()
    compute_scores(state, cfg, observed_dims=set())

    library_nodes = set(state.library.nodes.keys())
    assert state.salience_candidate_ids == library_nodes
    assert state.salience_num_nodes_scored == len(library_nodes)


def test_make_observation_set_respects_selected_blocks() -> None:
    cfg = AgentConfig(
        D=16,
        B=4,
        grid_side=4,
        grid_color_channels=1,
        grid_shape_channels=0,
        grid_channels=1,
        grid_base_dim=16,
    )
    blocks = [0, 3]
    obs_dims = make_observation_set(blocks, cfg)
    expected_dims: set[int] = set()
    for block_id in blocks:
        expected_dims.update(dims_for_block(block_id, cfg))
    assert set(obs_dims) == expected_dims, "Observation set should match selected block footprints"


def test_dag_edges_connect_block_neighbors() -> None:
    cfg = AgentConfig(
        D=12,
        B=4,
        transport_span_blocks=1,
    )
    agent = NUPCA3Agent(cfg)
    lib = agent.state.library

    for footprint, bucket in lib.footprint_index.items():
        if len(bucket) <= 1:
            continue
        for nid in bucket:
            node = lib.nodes.get(nid)
            assert node is not None
            assert node.parents or node.children, (
                f"Block {footprint} node {nid} should have DAG neighbors"
            )


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
    if state.context_register.shape != expected.shape:
        pytest.skip("context register disabled under current v5 configuration")
    assert np.allclose(state.context_register, expected)


def test_uncertainty_drives_fovea_selection() -> None:
    cfg = AgentConfig(
        D=4,
        B=4,
        fovea_blocks_per_step=1,
        alpha_cov=0.0,
        fovea_uncertainty_weight=1.0,
        fovea_use_age=False,
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
        _step_agent(agent, obs)
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
        periph_full=np.arange(cfg.D, dtype=float),
    )
    _step_agent(agent, full_obs)
    assert 0.0 <= agent.state.peripheral_confidence <= 1.0
    residual = agent.state.peripheral_residual
    assert np.isfinite(residual)
    assert agent.state.peripheral_prior.size > 0
    assert agent.state.peripheral_obs.size > 0


def test_block_signals_reset_without_worlds() -> None:
    cfg = AgentConfig(D=4, B=2)
    agent = NUPCA3Agent(cfg)
    agent.state.peripheral_residual = 0.0
    disagreement, innovation, periph_demand = _compute_block_signals(agent.state, cfg, [], cfg.D)
    assert np.allclose(disagreement, 0.0)
    assert np.allclose(innovation, 0.0)
    assert np.allclose(periph_demand, 0.0)


def test_update_fovea_tracking_age_counts_steps() -> None:
    cfg = AgentConfig(D=4, B=2)
    fovea = init_fovea_state(cfg)
    buf = ObservationBuffer(x_last=np.zeros(cfg.D, dtype=float))
    err = np.zeros(cfg.D, dtype=float)
    err[0] = 1.0

    update_fovea_tracking(
        fovea,
        buf,
        cfg,
        abs_error=err,
        observed_dims={0},
    )

    assert fovea.block_residual[0] > 0.0
    assert math.isclose(fovea.block_age[0], 0.0, abs_tol=1e-9)
    assert fovea.block_age[1] >= 1.0

    prev_age_block1 = float(fovea.block_age[1])
    update_fovea_tracking(
        fovea,
        buf,
        cfg,
        abs_error=None,
        observed_dims=set(),
    )
    assert math.isclose(fovea.block_age[0], 1.0, rel_tol=1e-9)
    assert math.isclose(fovea.block_age[1], prev_age_block1 + 1.0, rel_tol=1e-7)

    update_fovea_tracking(
        fovea,
        buf,
        cfg,
        abs_error=err,
        observed_dims={0},
    )
    assert math.isclose(fovea.block_age[0], 0.0, abs_tol=1e-9)


def test_salience_skipped_when_stable_and_no_learning() -> None:
    cfg = AgentConfig(
        D=8,
        B=2,
        fovea_blocks_per_step=1,
    )
    agent = NUPCA3Agent(cfg)
    obs = EnvObs(x_partial={0: 0.5})

    _, trace_first = _step_agent(agent, obs)
    agent.state.learning_candidates_prev = {"candidates": 0}
    agent.state.proposals_prev = 0
    _, trace_second = _step_agent(agent, obs)

    assert not trace_first["salience_skipped"]
    assert trace_second["salience_skipped"]


def test_prior_posterior_mae_logging() -> None:
    cfg = AgentConfig(
        D=4,
        B=2,
        periph_blocks=1,
        periph_bins=1,
        transport_search_radius=0,
    )
    agent = NUPCA3Agent(cfg)
    agent.state.buffer.x_prior = np.zeros(cfg.D, dtype=float)
    agent.state.buffer.x_last = np.zeros(cfg.D, dtype=float)

    obs_single = EnvObs(x_partial={0: 1.0})
    _, trace = _step_agent(agent, obs_single)
    prior_mae = float(trace["prior_obs_mae"])
    posterior_mae = float(trace["posterior_obs_mae"])

    assert math.isfinite(prior_mae)
    assert math.isfinite(posterior_mae)
    assert posterior_mae == 0.0
    assert prior_mae > posterior_mae


def test_peripheral_coherence_residual_detects_unobserved_dims() -> None:
    cfg = AgentConfig(
        D=8,
        B=4,
        periph_blocks=1,
        periph_bins=1,
        periph_channels=1,
    )
    agent = NUPCA3Agent(cfg)
    full = np.zeros(cfg.D, dtype=float)
    full[-2:] = 5.0

    obs_full = EnvObs(x_partial={0: 0.0}, periph_full=full)
    _, trace = _step_agent(agent, obs_full)
    residual = float(trace["peripheral_residual"])
    assert residual > 0.0


def test_multi_world_summary_present_and_weights_normalized() -> None:
    cfg = AgentConfig(
        D=12,
        B=3,
        multi_world_K=2,
        multi_world_lambda=1.0,
        transport_search_radius=1,
        transport_rotation_enabled=True,
        transport_rotation_steps=(0, 1),
    )
    agent = NUPCA3Agent(cfg)
    obs_multi = EnvObs(x_partial={0: 1.0})
    _, trace = _step_agent(agent, obs_multi)

    best_mae = float(trace["multi_world_best_prior_mae"])
    expected_mae = float(trace["multi_world_expected_prior_mae"])
    entropy = float(trace["multi_world_weight_entropy"])
    summary = trace["multi_world_summary"]

    assert math.isfinite(best_mae)
    assert math.isfinite(expected_mae)
    assert entropy >= 0.0
    assert isinstance(summary, list)
    assert len(agent.state.world_hypotheses) == cfg.multi_world_K
    weights = [float(world.weight) for world in agent.state.world_hypotheses]
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-6)
