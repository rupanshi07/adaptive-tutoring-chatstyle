"""
Module 3: Bayesian Network
Estimates P(Correct) from observable learner features.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def build_bayesian_network():
    model = DiscreteBayesianNetwork([
        ("Confidence", "Correct"),
        ("Difficulty", "Correct"),
        ("Time", "Correct"),
        ("Hints", "Correct"),
        ("PreviousAccuracy", "Correct"),
    ])

    cpd_confidence = TabularCPD("Confidence", 3, [[0.33], [0.34], [0.33]],
                                 state_names={"Confidence": ["Low", "Medium", "High"]})
    cpd_difficulty = TabularCPD("Difficulty", 3, [[0.34], [0.33], [0.33]],
                                  state_names={"Difficulty": ["Easy", "Medium", "Hard"]})
    cpd_time = TabularCPD("Time", 2, [[0.5], [0.5]],
                            state_names={"Time": ["Fast", "Slow"]})
    cpd_hints = TabularCPD("Hints", 2, [[0.7], [0.3]],
                             state_names={"Hints": ["No", "Yes"]})
    cpd_prev = TabularCPD("PreviousAccuracy", 3, [[0.33], [0.34], [0.33]],
                            state_names={"PreviousAccuracy": ["Low", "Medium", "High"]})

    num_combinations = 3 * 3 * 2 * 2 * 3
    correct_probs = []
    incorrect_probs = []

    import itertools
    combos = list(itertools.product(
        ["Low", "Medium", "High"],
        ["Easy", "Medium", "Hard"],
        ["Fast", "Slow"],
        ["No", "Yes"],
        ["Low", "Medium", "High"],
    ))

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
        "Correct", 2,
        [incorrect_probs, correct_probs],
        evidence=["Confidence", "Difficulty", "Time", "Hints", "PreviousAccuracy"],
        evidence_card=[3, 3, 2, 2, 3],
        state_names={
            "Correct": ["No", "Yes"],
            "Confidence": ["Low", "Medium", "High"],
            "Difficulty": ["Easy", "Medium", "Hard"],
            "Time": ["Fast", "Slow"],
            "Hints": ["No", "Yes"],
            "PreviousAccuracy": ["Low", "Medium", "High"],
        }
    )

    model.add_cpds(cpd_confidence, cpd_difficulty, cpd_time, cpd_hints, cpd_prev, cpd_correct)
    assert model.check_model()
    return model


def estimate_probability_correct(model, confidence, difficulty, time_, hints, previous_accuracy):
    infer = VariableElimination(model)
    result = infer.query(
        variables=["Correct"],
        evidence={
            "Confidence": confidence,
            "Difficulty": difficulty,
            "Time": time_,
            "Hints": hints,
            "PreviousAccuracy": previous_accuracy,
        }
    )
    return result.get_value(Correct="Yes")


if __name__ == "__main__":
    model = build_bayesian_network()
    p = estimate_probability_correct(model, "High", "Hard", "Fast", "No", "Low")
    print(f"P(Correct) = {p:.2f}")
