"""
Attempts to learn HMM transition and emission probabilities from real
logged interaction data (data/interaction_log.csv), using hmmlearn''s
Baum-Welch (EM) training, instead of the hand-specified matrices in
modules/hmm_calibration.py.

This script is honest about data sufficiency: it will refuse to train
and will explain why, rather than silently producing a meaningless model,
if there isn''t enough data or enough variety in the observations.
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import os

LOG_PATH = "data/interaction_log.csv"
MIN_ROWS = 50
MIN_DISTINCT_OBSERVATIONS = 4  # out of 6 possible (confidence x correctness)


def encode_observation(confidence_level, correct):
    conf_map = {"High": 0, "Medium": 1, "Low": 2}
    base = conf_map[confidence_level] * 2
    return base if correct else base + 1


def load_and_check_sufficiency():
    if not os.path.exists(LOG_PATH):
        return None, "No interaction log found yet. Run app.py and log real interactions first."

    df = pd.read_csv(LOG_PATH)
    df["correct_binary"] = df["correct"].astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "false": False, "0": False}
    )
    df = df.dropna(subset=["confidence", "correct_binary"])

    if len(df) < MIN_ROWS:
        return None, (
            f"Only {len(df)} logged interactions found; need at least {MIN_ROWS} "
            f"for Baum-Welch training to have any chance of producing a "
            f"meaningful (non-overfit) result. Training refused."
        )

    observations = [
        encode_observation(row["confidence"], row["correct_binary"])
        for _, row in df.iterrows()
    ]
    distinct = len(set(observations))

    if distinct < MIN_DISTINCT_OBSERVATIONS:
        return None, (
            f"Logged data only contains {distinct} distinct observation types "
            f"(out of 6 possible: combinations of Low/Medium/High confidence "
            f"x Correct/Incorrect). Training refused -- the model would only "
            f"ever have seen a narrow slice of possible learner behavior, "
            f"e.g. currently all logged confidence values may be the same "
            f"level, which cannot teach the model to distinguish states."
        )

    return observations, None


def train_hmm_from_data(observations):
    X = np.array(observations).reshape(-1, 1)
    lengths = [len(observations)]  # treated as one continuous sequence;
                                     # a real system should track session
                                     # boundaries per learner instead

    model = hmm.CategoricalHMM(n_components=3, n_iter=200, random_state=42)
    model.fit(X, lengths)
    return model


if __name__ == "__main__":
    observations, reason_refused = load_and_check_sufficiency()

    if reason_refused:
        print("TRAINING REFUSED -- insufficient data.")
        print(reason_refused)
        print("\nThe hand-specified HMM in modules/hmm_calibration.py will "
              "continue to be used until enough real, varied data accumulates.")
    else:
        print(f"Training on {len(observations)} real observations...")
        model = train_hmm_from_data(observations)
        print("\nLearned transition matrix:")
        print(model.transmat_)
        print("\nLearned emission matrix:")
        print(model.emissionprob_)
        print("\nLearned initial state distribution:")
        print(model.startprob_)
        print(
            "\nNOTE: learned component indices (0,1,2) are NOT automatically "
            "labeled Over-confident/Well-calibrated/Under-confident -- that "
            "mapping must be determined by inspecting which component''s "
            "emission distribution matches which semantic pattern, since "
            "Baum-Welch has no knowledge of our intended labels."
        )
