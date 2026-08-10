"""
Mathematical definition of learner calibration states.

Formalizes what "Over-confident", "Well-calibrated", and "Under-confident"
actually mean, independent of the HMM''s hand-specified emission matrix.
This gives a checkable, rule-based reference definition that the HMM''s
output can be validated against, and that a future data-learned HMM
should approximately reproduce.

Definition:
Let confidence be mapped to a numeric expectation of correctness:
    Low = 0.2, Medium = 0.5, High = 0.8
Let actual correctness be 1 (correct) or 0 (incorrect).

For a single interaction, the calibration error is:
    e_t = confidence_numeric(confidence_t) - actual_t

Over a rolling window of the last k interactions, the mean calibration
error is:
    E_k = mean(e_t for the last k interactions)

Classification rule (threshold = 0.15, chosen so that being consistently
one confidence level higher than warranted triggers a label change):
    E_k >  0.15   -> Over-confident   (stated confidence exceeds actual performance)
    E_k < -0.15   -> Under-confident  (stated confidence falls short of actual performance)
    otherwise     -> Well-calibrated  (stated confidence tracks actual performance)
"""

CONFIDENCE_NUMERIC = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
THRESHOLD = 0.15


def calibration_error(confidence_level, is_correct):
    """Single-interaction calibration error e_t, as defined above."""
    actual = 1.0 if is_correct else 0.0
    return CONFIDENCE_NUMERIC[confidence_level] - actual


def rolling_calibration_state(interactions, window=5):
    """
    interactions: list of (confidence_level, is_correct) tuples, most recent last.
    Returns the rule-based calibration state label using the last `window`
    interactions (or fewer, if not enough history exists yet).
    """
    if not interactions:
        return "Well-calibrated"  # explicit, documented cold-start default

    recent = interactions[-window:]
    errors = [calibration_error(conf, correct) for conf, correct in recent]
    mean_error = sum(errors) / len(errors)

    if mean_error > THRESHOLD:
        return "Over-confident"
    elif mean_error < -THRESHOLD:
        return "Under-confident"
    else:
        return "Well-calibrated"


if __name__ == "__main__":
    # Worked examples demonstrating the formal definition
    examples = [
        ("Consistently High confidence, mostly wrong",
         [("High", False), ("High", False), ("High", True)]),
        ("Consistently Low confidence, mostly right",
         [("Low", True), ("Low", True), ("Low", False)]),
        ("Confidence tracking actual performance",
         [("High", True), ("Low", False), ("Medium", True), ("Medium", False)]),
    ]

    for description, seq in examples:
        state = rolling_calibration_state(seq)
        errors = [round(calibration_error(c, r), 2) for c, r in seq]
        print(f"{description}")
        print(f"  Sequence: {seq}")
        print(f"  Per-step calibration errors: {errors}")
        print(f"  Mean error: {sum(errors)/len(errors):.3f}  ->  Rule-based state: {state}\n")

