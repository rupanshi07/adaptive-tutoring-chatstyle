"""
Dataset-driven Bayesian Network.
Learns CPDs from the real UCI Student Performance dataset using pgmpy''s
BayesianEstimator, instead of hand-specifying probabilities.

This complements the live interactive Bayesian Network (which must use
hand-specified priors, since confidence/hints are collected live and
have no equivalent in any historical dataset).
"""

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


def load_and_discretize():
    df = pd.read_csv("data/student_performance_raw.csv")

    df["Pass"] = df["G3"].apply(lambda x: "Yes" if x >= 10 else "No")

    def bucket_studytime(v):
        return "Low" if v <= 1 else ("Medium" if v == 2 else "High")

    def bucket_failures(v):
        return "None" if v == 0 else "One_or_more"

    def bucket_absences(v):
        return "Low" if v <= 4 else ("Medium" if v <= 10 else "High")

    df["StudyTime"] = df["studytime"].apply(bucket_studytime)
    df["Failures"] = df["failures"].apply(bucket_failures)
    df["Absences"] = df["absences"].apply(bucket_absences)

    return df[["StudyTime", "Failures", "Absences", "Pass"]]


def build_and_learn_model(data):
    model = DiscreteBayesianNetwork([
        ("StudyTime", "Pass"),
        ("Failures", "Pass"),
        ("Absences", "Pass"),
    ])

    estimator = BayesianEstimator(model, data)
    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=10,
    )
    model.add_cpds(*cpds)

    assert model.check_model()
    return model


def query_pass_probability(model, study_time, failures, absences):
    infer = VariableElimination(model)
    result = infer.query(
        variables=["Pass"],
        evidence={
            "StudyTime": study_time,
            "Failures": failures,
            "Absences": absences,
        },
    )
    return result.get_value(Pass="Yes")


if __name__ == "__main__":
    data = load_and_discretize()
    print(f"Loaded {len(data)} real student records.")
    print("\nSample of discretized data:")
    print(data.head(10))

    print("\nClass balance (Pass/Fail):")
    print(data["Pass"].value_counts())

    model = build_and_learn_model(data)
    print("\nModel learned successfully from real data.")

    print("\nLearned CPD for Pass (given StudyTime, Failures, Absences):")
    print(model.get_cpds("Pass"))

    print("\n--- Example queries using LEARNED (not hand-picked) probabilities ---")

    p1 = query_pass_probability(model, "High", "None", "Low")
    print(f"High study time, no failures, low absences -> P(Pass) = {p1:.2f}")

    p2 = query_pass_probability(model, "Low", "One_or_more", "High")
    print(f"Low study time, some failures, high absences -> P(Pass) = {p2:.2f}")


