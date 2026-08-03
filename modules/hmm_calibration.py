"""
Module 4: Hidden Markov Model
Infers hidden confidence calibration state from learner interaction history.
"""

import numpy as np
from hmmlearn import hmm

STATES = ["Over-confident", "Well-calibrated", "Under-confident"]


def build_hmm():
    model = hmm.CategoricalHMM(n_components=3, random_state=42, n_iter=100)

    # Initial state distribution
    model.startprob_ = np.array([0.34, 0.33, 0.33])

    # Transition matrix: calibration state tends to persist
    model.transmat_ = np.array([
        [0.7, 0.2, 0.1],   # Over-confident -> ...
        [0.15, 0.7, 0.15], # Well-calibrated -> ...
        [0.1, 0.2, 0.7],   # Under-confident -> ...
    ])

    # Emission probabilities: observation = combined code of
    # (confidence level 0-2) mismatched against (correctness 0-1)
    # We simplify to a single categorical observation with 6 symbols:
    # 0: HighConf+Correct, 1: HighConf+Incorrect,
    # 2: MedConf+Correct,  3: MedConf+Incorrect,
    # 4: LowConf+Correct,  5: LowConf+Incorrect
    model.emissionprob_ = np.array([
        [0.35, 0.30, 0.15, 0.10, 0.05, 0.05],  # Over-confident: high conf even when wrong
        [0.15, 0.10, 0.30, 0.15, 0.15, 0.15],  # Well-calibrated: confidence tracks correctness
        [0.05, 0.05, 0.10, 0.15, 0.30, 0.35],  # Under-confident: low conf even when right
    ])

    return model


def encode_observation(confidence_level, correct):
    """
    confidence_level: 'High', 'Medium', 'Low'
    correct: bool
    Returns integer symbol 0-5
    """
    conf_map = {"High": 0, "Medium": 1, "Low": 2}
    base = conf_map[confidence_level] * 2
    return base if correct else base + 1


def infer_calibration_state(model, observation_sequence):
    """
    observation_sequence: list of ints (0-5), one per past interaction
    Returns the most likely current calibration state (string)
    """
    X = np.array(observation_sequence).reshape(-1, 1)
    logprob, state_sequence = model.decode(X, algorithm="viterbi")
    current_state = state_sequence[-1]
    return STATES[current_state]


if __name__ == "__main__":
    model = build_hmm()

    # Example: learner answered confidently but incorrectly, twice in a row, then again
    history = [
        encode_observation("High", False),
        encode_observation("High", False),
        encode_observation("High", True),
    ]

    state = infer_calibration_state(model, history)
    print(f"Observation sequence: {history}")
    print(f"Inferred calibration state: {state}")
