"""
Attempts to relearn the live Bayesian Network''s Correct-node CPD from
real logged interactions (data/interaction_log.csv), using pgmpy''s
BayesianEstimator, instead of the hand-specified linear scoring formula
in modules/bayesian_network.py.

Uses effective_confidence (the blended self-reported + linguistic value
actually fed into the model at prediction time) as evidence -- NOT the
raw self-reported confidence, which would not match what the network
actually saw.

Like learn_hmm_from_log.py, this script is honest about data sufficiency:
it refuses to learn and explains why, rather than silently producing an
overfit or meaningless model, if there isn''t enough real, varied data.
"""

import pandas as pd
import os
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

LOG_PATH = "data/interaction_log.csv"
MIN_ROWS = 50
MIN_DISTINCT_PER_VARIABLE = 2  # each evidence variable should show at least 2 of its possible values


EVIDENCE_COLUMNS = {
    "effective_confidence": "Confidence",
    "difficulty": "Difficulty",
    "response_time": "Time",
    "hint_used": "Hints",
    "previous_accuracy": "PreviousAccuracy",
}


def load_and_check_sufficiency():
    if not os.path.exists(LOG_PATH):
        return None, "No interaction log found yet. Run app.py and log real interactions first."

    df = pd.read_csv(LOG_PATH)

    required_cols = list(EVIDENCE_COLUMNS.keys()) + ["correct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, (
            f"Log is missing required columns: {missing}. "
            f"This usually means the log was created before a schema change -- "
            f"rename or delete the old log and collect fresh data."
        )

    df = df.dropna(subset=required_cols)

    if len(df) < MIN_ROWS:
        return None, (
            f"Only {len(df)} usable logged interactions found; need at least "
            f"{MIN_ROWS} for parameter estimation to have any chance of being "
            f"meaningful rather than overfit. Learning refused."
        )

    insufficient_variety = []
    for col in EVIDENCE_COLUMNS:
        distinct = df[col].nunique()
        if distinct < MIN_DISTINCT_PER_VARIABLE:
            insufficient_variety.append(f"{col} (only {distinct} distinct value(s) seen)")

    if insufficient_variety:
        return None, (
            f"The following evidence variables do not have enough variety in "
            f"the logged data to learn a meaningful CPD: {insufficient_variety}. "
            f"Learning refused -- a network trained on this data could not "
            f"distinguish the effect of these variables from noise."
        )

    return df, None


def build_learned_network(df):
    df_renamed = df.rename(columns=EVIDENCE_COLUMNS)
    df_renamed["Correct"] = df_renamed["correct"].astype(str).str.strip().str.lower().map(
        {"true": "Yes", "1": "Yes", "false": "No", "0": "No"}
    )
    df_model = df_renamed[["Confidence", "Difficulty", "Time", "Hints", "PreviousAccuracy", "Correct"]].dropna()

    model = DiscreteBayesianNetwork([
        ("Confidence", "Correct"), ("Difficulty", "Correct"), ("Time", "Correct"),
        ("Hints", "Correct"), ("PreviousAccuracy", "Correct"),
    ])

    estimator = BayesianEstimator(model, df_model)
    cpd_correct = estimator.estimate_cpd("Correct", prior_type="BDeu", equivalent_sample_size=10)
    model.add_cpds(cpd_correct)

    for parent_var, states in [
        ("Confidence", df_model["Confidence"].unique()),
        ("Difficulty", df_model["Difficulty"].unique()),
        ("Time", df_model["Time"].unique()),
        ("Hints", df_model["Hints"].unique()),
        ("PreviousAccuracy", df_model["PreviousAccuracy"].unique()),
    ]:
        parent_cpd = estimator.estimate_cpd(parent_var, prior_type="BDeu", equivalent_sample_size=10)
        model.add_cpds(parent_cpd)

    assert model.check_model()
    return model, len(df_model)


if __name__ == "__main__":
    df, reason_refused = load_and_check_sufficiency()

    if reason_refused:
        print("LEARNING REFUSED -- insufficient data.")
        print(reason_refused)
        print("\nThe hand-specified Correct-node CPD in modules/bayesian_network.py "
              "will continue to be used until enough real, varied data accumulates.")
    else:
        model, n_used = build_learned_network(df)
        print(f"Learned from {n_used} real logged interactions.\n")
        print("Learned CPD for Correct:")
        print(model.get_cpds("Correct"))

        infer = VariableElimination(model)
        print("\nCompare to the hand-specified formula for the same evidence "
              "(High confidence, Medium difficulty, Fast, No hint, High previous accuracy):")
        try:
            result = infer.query(
                variables=["Correct"],
                evidence={
                    "Confidence": "High", "Difficulty": "Medium", "Time": "Fast",
                    "Hints": "No", "PreviousAccuracy": "High",
                },
            )
            print(result)
        except Exception as e:
            print(f"Could not query this exact combination from real data yet: {e}")
