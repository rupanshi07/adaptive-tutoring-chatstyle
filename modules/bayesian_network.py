"""
Module 3: Bayesian Network (redesigned)

Structure now captures real dependencies between variables, instead of
treating them as independent causes fanning into one child (the "naive
Bayes"-like structure this replaces):

    PreviousAccuracy --------> Confidence
    Difficulty -------------> Hints
    Difficulty, Hints -------> Time
    Confidence, Difficulty, Time, Hints, PreviousAccuracy -> Correct

Every edge is causally motivated: harder questions increase hint usage
and response time; using a hint adds extra time; prior performance
plausibly shapes stated confidence.
"""

import itertools
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def build_bayesian_network():
    model = DiscreteBayesianNetwork([
        ("PreviousAccuracy", "Confidence"),
        ("Difficulty", "Hints"),
        ("Difficulty", "Time"),
        ("Hints", "Time"),
        ("Confidence", "Correct"),
        ("Difficulty", "Correct"),
        ("Time", "Correct"),
        ("Hints", "Correct"),
        ("PreviousAccuracy", "Correct"),
    ])

    # ---- Root nodes: near-uniform priors ----
    cpd_previous = TabularCPD("PreviousAccuracy", 3, [[0.33], [0.34], [0.33]],
                               state_names={"PreviousAccuracy": ["Low", "Medium", "High"]})
    cpd_difficulty = TabularCPD("Difficulty", 3, [[0.34], [0.33], [0.33]],
                                 state_names={"Difficulty": ["Easy", "Medium", "Hard"]})

    # ---- Confidence depends on PreviousAccuracy ----
    # A stronger track record plausibly shifts stated confidence upward.
    conf_values = {"Low": [], "Medium": [], "High": []}
    for prev in ["Low", "Medium", "High"]:
        dist = {"Low": {"Low": 0.50, "Medium": 0.35, "High": 0.15},
                "Medium": {"Low": 0.25, "Medium": 0.50, "High": 0.25},
                "High": {"Low": 0.15, "Medium": 0.35, "High": 0.50}}[prev]
        for level in ["Low", "Medium", "High"]:
            conf_values[level].append(dist[level])
    cpd_confidence = TabularCPD(
        "Confidence", 3,
        [conf_values["Low"], conf_values["Medium"], conf_values["High"]],
        evidence=["PreviousAccuracy"], evidence_card=[3],
        state_names={"Confidence": ["Low", "Medium", "High"], "PreviousAccuracy": ["Low", "Medium", "High"]},
    )

    # ---- Hints depend on Difficulty ----
    # Harder questions make hint use more likely.
    hint_prob_yes = {"Easy": 0.10, "Medium": 0.30, "Hard": 0.50}
    no_row, yes_row = [], []
    for diff in ["Easy", "Medium", "Hard"]:
        yes_row.append(hint_prob_yes[diff])
        no_row.append(1 - hint_prob_yes[diff])
    cpd_hints = TabularCPD(
        "Hints", 2, [no_row, yes_row],
        evidence=["Difficulty"], evidence_card=[3],
        state_names={"Hints": ["No", "Yes"], "Difficulty": ["Easy", "Medium", "Hard"]},
    )

    # ---- Time depends on Difficulty AND Hints ----
    # Harder questions take longer; using a hint adds extra time.
    fast_prob = {
        ("Easy", "No"): 0.80, ("Easy", "Yes"): 0.60,
        ("Medium", "No"): 0.60, ("Medium", "Yes"): 0.40,
        ("Hard", "No"): 0.40, ("Hard", "Yes"): 0.20,
    }
    fast_row, slow_row = [], []
    for diff, hint in itertools.product(["Easy", "Medium", "Hard"], ["No", "Yes"]):
        p_fast = fast_prob[(diff, hint)]
        fast_row.append(p_fast)
        slow_row.append(1 - p_fast)
    cpd_time = TabularCPD(
        "Time", 2, [fast_row, slow_row],
        evidence=["Difficulty", "Hints"], evidence_card=[3, 2],
        state_names={"Time": ["Fast", "Slow"], "Difficulty": ["Easy", "Medium", "Hard"], "Hints": ["No", "Yes"]},
    )

    # ---- Correct depends on all five (same scoring formula as before) ----
    combos = list(itertools.product(
        ["Low", "Medium", "High"], ["Easy", "Medium", "Hard"],
        ["Fast", "Slow"], ["No", "Yes"], ["Low", "Medium", "High"],
    ))
    correct_probs, incorrect_probs = [], []
    for conf, diff, time_, hint, prev in combos:
        score = 0.5
        score += {"Low": -0.15, "Medium": 0.0, "High": 0.15}[conf]
        score += {"Easy": 0.2, "Medium": 0.0, "Hard": -0.2}[diff]
        score += {"Fast": 0.05, "Slow": -0.05}[time_]
        score += {"No": 0.0, "Yes": -0.1}[hint]
        score += {"Low": -0.15, "Medium": 0.0, "High": 0.15}[prev]
        score = min(max(score, 0.02), 0.98)
        correct_probs.append(score)
        incorrect_probs.append(1 - score)

    cpd_correct = TabularCPD(
        "Correct", 2, [incorrect_probs, correct_probs],
        evidence=["Confidence", "Difficulty", "Time", "Hints", "PreviousAccuracy"],
        evidence_card=[3, 3, 2, 2, 3],
        state_names={
            "Correct": ["No", "Yes"], "Confidence": ["Low", "Medium", "High"],
            "Difficulty": ["Easy", "Medium", "Hard"], "Time": ["Fast", "Slow"],
            "Hints": ["No", "Yes"], "PreviousAccuracy": ["Low", "Medium", "High"],
        },
    )

    model.add_cpds(cpd_previous, cpd_difficulty, cpd_confidence, cpd_hints, cpd_time, cpd_correct)
    assert model.check_model()
    return model


def estimate_probability_correct(model, confidence, difficulty, time_, hints, previous_accuracy):
    infer = VariableElimination(model)
    result = infer.query(
        variables=["Correct"],
        evidence={
            "Confidence": confidence, "Difficulty": difficulty,
            "Time": time_, "Hints": hints, "PreviousAccuracy": previous_accuracy,
        },
    )
    return result.get_value(Correct="Yes")


if __name__ == "__main__":
    model = build_bayesian_network()
    print("Model structure (edges):")
    for edge in model.edges():
        print(f"  {edge[0]} -> {edge[1]}")

    p = estimate_probability_correct(model, "High", "Hard", "Fast", "No", "Low")
    print(f"\nP(Correct) = {p:.2f}  (should still be 0.35, same as before -- structure changed, Correct formula did not)")

    infer = VariableElimination(model)
    partial = infer.query(variables=["Time"], evidence={"Difficulty": "Hard", "Hints": "Yes"})
    print(f"\nNEW capability -- inferring Time from partial evidence (Difficulty=Hard, Hints=Yes):")
    print(partial)
