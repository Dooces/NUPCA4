import math
import os
import sys
import time

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.edits.acceptance import check_permit_struct
from nupca3.geometry.fovea import make_observation_set, select_fovea
from nupca3.memory.completion import complete
from nupca3.memory.fusion import fuse_predictions
from nupca3.memory.audit import audit_sig64_index_health
from nupca3.memory.working_set import get_retrieval_candidates
from nupca3.types import EnvObs, ExpertNode, LearningCache, Stress, WorkingSet


def _step_agent(agent: NUPCA3Agent, obs: EnvObs):
    obs.t_w = agent.state.t_w + 1
    obs.wall_ms = int(time.perf_counter() * 1000)
    return agent.step(obs)


def test_completion_operator_unified_modes() -> None:
    cfg = AgentConfig(D=8, B=4, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)

    prior = np.arange(cfg.D, dtype=float)
    sigma_diag = np.ones(cfg.D, dtype=float)
    cue = {0: 10.0, 3: -1.0}

    x_per, sigma_per, prior_per = complete(
        cue,
        mode="perception",
        state=agent.state,
        cfg=cfg,
        predicted_prior_t=prior,
        predicted_sigma_diag=sigma_diag,
    )
    x_rec, sigma_rec, prior_rec = complete(
        cue,
        mode="recall",
        state=agent.state,
        cfg=cfg,
        predicted_prior_t=prior,
        predicted_sigma_diag=sigma_diag,
    )
    x_pred, sigma_pred, prior_pred = complete(
        cue,
        mode="prediction",
        state=agent.state,
        cfg=cfg,
        predicted_prior_t=prior,
        predicted_sigma_diag=sigma_diag,
    )

    assert np.allclose(prior_per, prior)
    assert np.allclose(prior_rec, prior)
    assert np.allclose(prior_pred, prior)

    assert np.allclose(x_per, x_rec)
    assert np.allclose(x_pred, prior)

    for k, v in cue.items():
        assert float(x_per[int(k)]) == float(v)
    for k in range(cfg.D):
        if k not in cue:
            assert float(x_per[k]) == float(prior[k])

    assert np.allclose(np.diag(sigma_per), sigma_diag)
    assert np.allclose(np.diag(sigma_rec), sigma_diag)
    assert np.allclose(np.diag(sigma_pred), sigma_diag)


def test_step_pipeline_respects_fovea_mask() -> None:
    cfg = AgentConfig(D=8, B=4, fovea_blocks_per_step=1, coverage_cap_G=0)
    agent = NUPCA3Agent(cfg)

    full_obs = EnvObs(
        x_partial={i: float(i + 1) for i in range(cfg.D)},
        opp=0.0,
        danger=0.0,
    )

    _step_agent(agent, full_obs)

    current_blocks = getattr(agent.state.fovea, "current_blocks", set()) or set()
    expected_obs = make_observation_set(current_blocks, cfg)

    assert agent.state.buffer.observed_dims == expected_obs
    for k in expected_obs:
        assert float(agent.state.buffer.x_last[k]) == float(full_obs.x_partial[k])


def test_fusion_coverage_invariant_no_active() -> None:
    cfg = AgentConfig(D=6, B=3, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)
    agent.state.buffer.x_last = np.arange(cfg.D, dtype=float)

    empty = WorkingSet(active=[])
    x_hat, Sigma = fuse_predictions(agent.state.library, empty, agent.state.buffer, set(), cfg)

    assert np.allclose(x_hat, agent.state.buffer.x_last)
    assert np.all(np.isinf(np.diag(Sigma)))


def test_permit_struct_requires_rest_and_stability() -> None:
    cfg = AgentConfig(D=4, B=2, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)

    agent.state.macro.rest = False
    agent.state.arousal_prev = 0.0
    agent.state.probe_var = 0.0
    agent.state.feature_var = 0.0
    assert check_permit_struct(agent.state, cfg) is False

    agent.state.macro.rest = True
    agent.state.t_w = 1000
    agent.state.probe_var = 0.0
    agent.state.feature_var = 0.0
    assert check_permit_struct(agent.state, cfg) is True


def _find_anchor_by_mask(lib, on_dim: int, off_dim: int):
    for node in lib.nodes.values():
        mask = np.asarray(node.mask, dtype=float).reshape(-1)
        if int(on_dim) < mask.size and int(off_dim) < mask.size:
            if mask[int(on_dim)] > 0.5 and mask[int(off_dim)] <= 0.5:
                return node
    return None


def test_responsibility_gating_updates_only_observed_block() -> None:
    cfg = AgentConfig(
        D=4,
        B=2,
        fovea_blocks_per_step=1,
        coverage_cap_G=0,
        theta_learn=0.1,
        lr_expert=0.1,
    )
    agent = NUPCA3Agent(cfg)

    agent.state.buffer.x_last = np.ones(cfg.D, dtype=float)
    agent.state.learn_cache = LearningCache(
        x_t=np.zeros(cfg.D, dtype=float),
        yhat_tp1=np.zeros(cfg.D, dtype=float),
        sigma_tp1_diag=np.ones(cfg.D, dtype=float),
        A_t=WorkingSet(active=[]),
        permit_param_t=True,
        rest_t=False,
    )

    anchor_b0 = _find_anchor_by_mask(agent.state.library, on_dim=0, off_dim=2)
    anchor_b1 = _find_anchor_by_mask(agent.state.library, on_dim=2, off_dim=0)
    assert anchor_b0 is not None and anchor_b1 is not None

    w0_before = anchor_b0.W.copy()
    b0_before = anchor_b0.b.copy()
    w1_before = anchor_b1.W.copy()
    b1_before = anchor_b1.b.copy()

    obs = EnvObs(x_partial={0: 0.05, 1: 0.05, 2: 0.9, 3: 0.9})
    _step_agent(agent, obs)

    observed_blocks = {int(b) for b in getattr(agent.state.fovea, "current_blocks", set()) or set()}
    assert observed_blocks, "Expected at least one observed block"

    def _updated(node, before_W, before_b):
        return not (np.allclose(node.W, before_W) and np.allclose(node.b, before_b))

    updated_b0 = _updated(anchor_b0, w0_before, b0_before)
    updated_b1 = _updated(anchor_b1, w1_before, b1_before)

    assert updated_b0 == (0 in observed_blocks)
    assert updated_b1 == (1 in observed_blocks)


def test_responsibility_gating_respects_error_threshold() -> None:
    cfg = AgentConfig(
        D=4,
        B=2,
        fovea_blocks_per_step=1,
        coverage_cap_G=0,
        theta_learn=0.01,
        lr_expert=0.1,
    )
    agent = NUPCA3Agent(cfg)

    agent.state.buffer.x_last = np.ones(cfg.D, dtype=float)
    agent.state.learn_cache = LearningCache(
        x_t=np.zeros(cfg.D, dtype=float),
        yhat_tp1=np.zeros(cfg.D, dtype=float),
        sigma_tp1_diag=np.ones(cfg.D, dtype=float),
        A_t=WorkingSet(active=[]),
        permit_param_t=True,
        rest_t=False,
    )

    anchor_b0 = _find_anchor_by_mask(agent.state.library, on_dim=0, off_dim=2)
    assert anchor_b0 is not None

    w_before = anchor_b0.W.copy()
    b_before = anchor_b0.b.copy()

    obs = EnvObs(x_partial={0: 0.05, 1: 0.05, 2: 0.9, 3: 0.9})
    _step_agent(agent, obs)

    assert np.allclose(anchor_b0.W, w_before)
    assert np.allclose(anchor_b0.b, b_before)


def test_retrieval_keyed_to_greedy_cov_blocks() -> None:
    cfg = AgentConfig(
        D=4,
        B=2,
        fovea_blocks_per_step=1,
        coverage_cap_G=0,
        fovea_use_age=False,
    )
    agent = NUPCA3Agent(cfg)

    # Block 1 should dominate greedy_cov selection.
    agent.state.fovea.block_residual = np.array([0.1, 2.0], dtype=float)
    agent.state.fovea.block_age = np.array([0, 0], dtype=int)
    agent.state.fovea.current_blocks = set()

    mask_b0 = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
    mask_b1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    node_b0 = ExpertNode(
        node_id=10,
        mask=mask_b0,
        W=np.eye(cfg.D),
        b=np.zeros(cfg.D),
        Sigma=np.eye(cfg.D),
        reliability=0.5,
        cost=1.0,
        is_anchor=False,
        footprint=0,
        unit_sig64=1,
    )
    node_b1 = ExpertNode(
        node_id=11,
        mask=mask_b1,
        W=np.eye(cfg.D),
        b=np.zeros(cfg.D),
        Sigma=np.eye(cfg.D),
        reliability=0.5,
        cost=1.0,
        is_anchor=False,
        footprint=1,
        unit_sig64=2,
    )
    agent.state.library.add_node(node_b0)
    agent.state.library.add_node(node_b1)
    agent.state.last_sig64 = node_b1.unit_sig64
    agent.state.fovea.current_blocks = {1}

    agent.state.incumbents_by_block = [{node_b0.node_id}, {node_b1.node_id}]
    agent.state.incumbents_revision = int(agent.state.library.revision)
    agent.state.active_set = set()

    retrieved = get_retrieval_candidates(agent.state, cfg)
    assert node_b1.node_id in retrieved
    assert node_b0.node_id not in retrieved


def test_retrieval_streaming_topk_enforces_caps_and_ties() -> None:
    cfg = AgentConfig(
        D=4,
        B=2,
        fovea_blocks_per_step=1,
        B_max=4,
        F_max=2,
        C_cand_max=3,
        K_max=2,
        sig_query_cand_cap=5,
        max_retrieval_candidates=5,
    )
    agent = NUPCA3Agent(cfg)
    agent.state.fovea.current_blocks = {0}
    base_sig = 0b1111
    agent.state.last_sig64 = base_sig
    agent.state.active_set = set()

    class StubSigIndex:
        def __init__(self, out: list[int]):
            self.out = out
            self.calls: list[int] = []
            self.tables = 1
            self.bucket_bits = 1
            self.bucket_cap = 1
            self.n_buckets = 2
            self.err_bins = 3
            self.buckets = [dict()]

        def query(self, sig64: int, block_ids, cand_cap: int = 0):
            self.calls.append(int(cand_cap))
            return list(self.out)

        def get_error(self, node_id: int, h_bin: int) -> float:
            return 0.0

    stub = StubSigIndex([1, 2, 3, 4])
    agent.state.library.sig_index = stub
    agent.state.library.nodes = {}

    for nid in range(1, 5):
        node = ExpertNode(
            node_id=nid,
            mask=np.ones(cfg.D),
            W=np.eye(cfg.D),
            b=np.zeros(cfg.D),
            Sigma=np.eye(cfg.D),
            reliability=1.0,
            cost=1.0,
            is_anchor=False,
            footprint=0,
            unit_sig64=base_sig if nid <= 3 else base_sig ^ 1,
        )
        agent.state.library.nodes[nid] = node

    retrieved = get_retrieval_candidates(agent.state, cfg)
    assert stub.calls and max(stub.calls) <= cfg.C_cand_max
    assert retrieved == {1, 2}


def test_sig64_metadata_respects_configured_caps() -> None:
    cfg = AgentConfig(D=4, B=2, B_max=16, fovea_blocks_per_step=1, F_max=3)
    agent = NUPCA3Agent(cfg)

    obs = EnvObs(x_partial={0: 0.5}, periph_full=np.zeros(cfg.D), pos_dims={0})
    _step_agent(agent, obs)

    expected_mask_bytes = max(1, (cfg.B_max + 7) // 8)
    assert agent.state.sig_prev_counts.shape[0] == cfg.F_max
    assert agent.state.sig_prev_hist.size == expected_mask_bytes
    assert audit_sig64_index_health(agent.state, cfg) == []


def test_rest_state_uses_lagged_predicates() -> None:
    cfg = AgentConfig(D=4, B=2, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)

    agent.state.rest_permitted_prev = True
    agent.state.demand_prev = True
    agent.state.interrupt_prev = False

    obs = EnvObs(x_partial={0: 0.0})
    _action, trace = _step_agent(agent, obs)
    assert trace["rest"] is True

    agent.state.rest_permitted_prev = True
    agent.state.demand_prev = False
    agent.state.interrupt_prev = False

    _action, trace = _step_agent(agent, obs)
    assert trace["rest"] is False


def test_permit_param_uses_lagged_signals() -> None:
    cfg = AgentConfig(D=4, B=2, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)

    agent.state.rest_permitted_prev = True
    agent.state.demand_prev = False
    agent.state.interrupt_prev = False
    agent.state.stress = Stress(s_E=0.0, s_D=0.0, s_L=0.0, s_C=0.0, s_S=0.0, s_int_need=0.0, s_ext_th=0.0)

    agent.state.x_C_prev = 10.0
    agent.state.arousal_prev = 0.0
    agent.state.rawE_prev = 1.0
    agent.state.rawD_prev = 1.0

    obs = EnvObs(x_partial={0: 0.0})
    _action, trace = _step_agent(agent, obs)
    assert trace["permit_param"] is True

    agent.state.x_C_prev = -1.0
    _action, trace = _step_agent(agent, obs)
    assert trace["permit_param"] is False


def test_rest_permission_requires_stability_window() -> None:
    cfg = AgentConfig(D=4, B=2, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)

    agent.state.rest_permitted_prev = True
    agent.state.demand_prev = True
    agent.state.interrupt_prev = False
    agent.state.probe_window = []
    agent.state.feature_window = []

    obs = EnvObs(x_partial={0: 0.0})
    _action, trace = _step_agent(agent, obs)

    assert trace["rest_permitted_t"] is False


def test_margins_use_opportunity_signal() -> None:
    cfg = AgentConfig(D=4, B=2, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)

    opp = 0.37
    obs = EnvObs(x_partial={0: 0.0}, opp=opp, danger=0.0)
    _step_agent(agent, obs)

    assert abs(agent.state.margins.m_L - opp) < 1e-9


def test_spawn_proposals_enqueue_in_operating() -> None:
    cfg = AgentConfig(
        D=4,
        B=2,
        fovea_blocks_per_step=1,
        coverage_cap_G=0,
        theta_spawn=0.01,
        K=1,
    )
    agent = NUPCA3Agent(cfg)

    agent.state.fovea.block_residual = np.array([1.0, 0.0], dtype=float)
    agent.state.fovea.block_age = np.array([0, 0], dtype=int)

    obs = EnvObs(x_partial={0: 1.0, 1: 1.0})
    _action, trace = _step_agent(agent, obs)

    assert trace["rest"] is False
    assert trace["Q_struct_len"] >= 1


def test_salience_trace_never_full_library() -> None:
    cfg = AgentConfig(D=8, B=4, fovea_blocks_per_step=1)
    agent = NUPCA3Agent(cfg)
    obs = EnvObs(x_partial={0: 1.0})

    _action, trace = _step_agent(agent, obs)
    library_size = int(trace["salience_library_size"])
    candidate_count = int(trace["salience_candidate_count"])
    nodes_scored = int(trace["salience_nodes_scored"])
    ratio = float(trace["salience_candidate_ratio"])

    assert library_size > 0
    assert 0 < candidate_count < library_size
    assert nodes_scored == candidate_count
    expected_ratio = candidate_count / library_size
    assert math.isclose(ratio, expected_ratio, rel_tol=1e-9)


def test_salience_trace_debug_exhaustive_includes_full_library() -> None:
    cfg = AgentConfig(
        D=8,
        B=4,
        fovea_blocks_per_step=1,
        salience_debug_exhaustive=True,
    )
    agent = NUPCA3Agent(cfg)
    obs = EnvObs(x_partial={0: 1.0})

    _action, trace = _step_agent(agent, obs)
    library_size = int(trace["salience_library_size"])
    candidate_count = int(trace["salience_candidate_count"])
    nodes_scored = int(trace["salience_nodes_scored"])
    ratio = float(trace["salience_candidate_ratio"])

    assert library_size > 0
    assert candidate_count == library_size
    assert nodes_scored == library_size
    assert math.isclose(ratio, 1.0, rel_tol=1e-9)
