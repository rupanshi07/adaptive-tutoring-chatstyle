"""Unit + integration tests for the adaptive tutoring pipeline (no API keys needed).
Run from the project root:  python -m pytest tests -v
"""
import numpy as np
import pytest

from modules.bayesian_network import (
    build_bayesian_network, estimate_probability_correct,
    estimate_preliminary_probability_correct)
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, bucket_probability, compute_reward, ACTIONS, STATE_SPACE
from modules.tutor import blend_confidence, grade_answer
from hmm_state_definition import rolling_calibration_state, calibration_error


@pytest.fixture(scope="module")
def bn():
    return build_bayesian_network()


# ---------------- Bayesian network ----------------
def test_bn_is_valid_dag_with_expected_structure(bn):
    assert bn.check_model()
    assert len(bn.nodes()) == 6 and len(bn.edges()) == 9
    assert set(bn.get_parents("Correct")) == {"Confidence", "Difficulty", "Time", "Hints", "PreviousAccuracy"}


def test_bn_known_values(bn):
    assert estimate_probability_correct(bn, "High", "Hard", "Fast", "No", "Low") == pytest.approx(0.35)
    assert estimate_probability_correct(bn, "High", "Easy", "Fast", "No", "High") == pytest.approx(0.98)  # clipped
    assert estimate_probability_correct(bn, "Low", "Hard", "Slow", "Yes", "Low") == pytest.approx(0.02)  # clipped


def test_bn_monotonic_in_difficulty_and_history(bn):
    p = [estimate_preliminary_probability_correct(bn, d, "Medium") for d in ("Easy", "Medium", "Hard")]
    assert p[0] > p[1] > p[2]
    q = [estimate_preliminary_probability_correct(bn, "Medium", a) for a in ("Low", "Medium", "High")]
    assert q[0] < q[1] < q[2]


def test_hint_lowers_probability_given_same_evidence(bn):
    no = estimate_probability_correct(bn, "Medium", "Medium", "Fast", "No", "Medium")
    yes = estimate_probability_correct(bn, "Medium", "Medium", "Fast", "Yes", "Medium")
    assert yes < no


def test_d_separation_claims(bn):
    assert not bn.is_dconnected("Confidence", "Difficulty")
    assert not bn.is_dconnected("Time", "PreviousAccuracy", observed=["Difficulty", "Hints"])
    assert bn.is_dconnected("Hints", "Time")


def test_partial_evidence_marginalises_unobserved(bn):
    p = estimate_preliminary_probability_correct(bn, "Hard", "Low")
    assert 0 < p < 1 and p == pytest.approx(0.078, abs=0.002)


# ---------------- HMM ----------------
def test_hmm_matrices_are_stochastic():
    h = build_hmm()
    assert np.allclose(h.transmat_.sum(axis=1), 1) and np.allclose(h.emissionprob_.sum(axis=1), 1)
    assert h.startprob_.sum() == pytest.approx(1)


def test_hmm_state_inference_examples():
    h = build_hmm()
    over = [encode_observation("High", False), encode_observation("High", False), encode_observation("High", True)]
    under = [encode_observation("Low", True)] * 3
    assert infer_calibration_state(h, over) == "Over-confident"
    assert infer_calibration_state(h, under) == "Under-confident"


def test_encode_observation_range():
    codes = {encode_observation(c, r) for c in ("High", "Medium", "Low") for r in (True, False)}
    assert codes == set(range(6))


def test_rule_based_definition_agrees_with_hmm_on_examples():
    h = build_hmm()
    seq = [("High", False), ("High", False), ("High", True)]
    obs = [encode_observation(c, r) for c, r in seq]
    assert rolling_calibration_state(seq) == infer_calibration_state(h, obs) == "Over-confident"
    assert calibration_error("High", False) == pytest.approx(0.8)


# ---------------- RL agent ----------------
def test_bucket_boundaries():
    assert [bucket_probability(x) for x in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0)] == \
           ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_reward_mapping():
    assert compute_reward("improved_after_hint") == 10 and compute_reward("quit") == -10
    assert compute_reward("unknown") == 0


def test_q_update_follows_bellman_rule(tmp_path):
    a = TutorRLAgent(alpha=0.5, gamma=0.9, epsilon=0.0, q_table_path=str(tmp_path / "q.json"))
    s, s2 = STATE_SPACE[0], STATE_SPACE[4]
    a.q_table[s2][:] = 10.0
    a.update(s, "Hint", 10, s2)
    assert a.q_table[s][ACTIONS.index("Hint")] == pytest.approx(0.5 * (10 + 0.9 * 10))


def test_q_table_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "q.json")
    a = TutorRLAgent(q_table_path=path); a.q_table[STATE_SPACE[1]][2] = 3.5; a.save_q_table()
    b = TutorRLAgent(q_table_path=path)
    assert b.q_table[STATE_SPACE[1]][2] == pytest.approx(3.5)


def test_greedy_agent_picks_argmax(tmp_path):
    a = TutorRLAgent(epsilon=0.0, q_table_path=str(tmp_path / "q.json"))
    a.q_table[STATE_SPACE[0]] = np.array([0.0, 1.0, 5.0, 2.0])
    assert a.choose_action(STATE_SPACE[0]) == "Retry"


# ---------------- tutor helpers (offline) ----------------
def test_blend_confidence():
    assert blend_confidence("High", None) == "High"
    assert blend_confidence("High", "Low") in ("Medium",)
    assert blend_confidence("Low", "Low") == "Low"


def test_mcq_grading_offline():
    q = {"question_type": "MCQ", "correct_option": "B"}
    assert grade_answer(q, " b ") == (True, False)
    assert grade_answer(q, "A") == (False, False)


# ---------------- integration: full pipeline without LLM ----------------
def test_pipeline_bn_hmm_rl_end_to_end(bn, tmp_path):
    h = build_hmm()
    agent = TutorRLAgent(epsilon=0.0, q_table_path=str(tmp_path / "q.json"))
    p = estimate_probability_correct(bn, "High", "Hard", "Fast", "No", "Low")
    state = infer_calibration_state(h, [encode_observation("High", False)])
    action = agent.choose_action((bucket_probability(p), state))
    assert action in ACTIONS
