"""
Rigorous RL evaluation: trained Q-learning agent vs. a random policy vs. a
fixed "always Hint" policy, across multiple independent trials, reporting
mean and standard deviation -- not a single run.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, ACTIONS, compute_reward, bucket_probability

DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFFICULTY_BIAS = {"Easy": 0.75, "Medium": 0.55, "Hard": 0.35}

NUM_TRIALS = 10
EPISODES_PER_TRIAL = 500


def simulate_step(bn_model, hmm_model, agent, learner_history, confidence_level,
                   previous_accuracy, policy="trained"):
    difficulty = random.choice(DIFFICULTIES)
    is_correct = random.random() < DIFFICULTY_BIAS[difficulty]
    time_taken = random.choice(["Fast", "Slow"])

    p_correct = estimate_probability_correct(
        bn_model, confidence=confidence_level, difficulty=difficulty,
        time_=time_taken, hints="No", previous_accuracy=previous_accuracy,
    )

    observation = encode_observation(confidence_level, is_correct)
    updated_history = learner_history + [observation]
    if len(updated_history) > 20:
        updated_history = updated_history[-20:]
    calibration_state = infer_calibration_state(hmm_model, updated_history)

    prob_bucket = bucket_probability(p_correct)
    rl_state = (prob_bucket, calibration_state)

    if policy == "trained":
        action = agent.choose_action(rl_state)
    elif policy == "random":
        action = random.choice(ACTIONS)
    elif policy == "fixed_hint":
        action = "Hint"
    else:
        raise ValueError(f"Unknown policy: {policy}")

    outcome = "needed_solution" if action == "Reveal" else (
        "improved_after_hint" if is_correct else "repeated_mistake"
    )
    reward = compute_reward(outcome)

    return updated_history, is_correct, reward


def run_trial(bn_model, hmm_model, agent, policy, num_episodes):
    learner_history = []
    previous_accuracy = "Medium"
    total = 0

    for _ in range(num_episodes):
        confidence_level = random.choice(["Low", "Medium", "High"])
        learner_history, is_correct, reward = simulate_step(
            bn_model, hmm_model, agent, learner_history, confidence_level,
            previous_accuracy, policy=policy,
        )
        total += reward
        previous_accuracy = "High" if is_correct else "Low"

    return total


def run_multi_trial_evaluation():
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()

    results = {"trained": [], "random": [], "fixed_hint": []}

    for trial in range(NUM_TRIALS):
        trained_agent = TutorRLAgent(epsilon=0.0)  # loads the persisted, already-trained Q-table

        for policy in ["trained", "random", "fixed_hint"]:
            total_reward = run_trial(bn_model, hmm_model, trained_agent, policy, EPISODES_PER_TRIAL)
            results[policy].append(total_reward)

        print(f"Trial {trial + 1}/{NUM_TRIALS} complete "
              f"(trained={results['trained'][-1]}, "
              f"random={results['random'][-1]}, "
              f"fixed_hint={results['fixed_hint'][-1]})")

    return results


def summarize_and_plot(results, save_path="plots/rl_multi_trial_comparison.png"):
    print("\n" + "=" * 60)
    print(f"Summary over {NUM_TRIALS} independent trials, "
          f"{EPISODES_PER_TRIAL} episodes each:")
    print("=" * 60)

    summary = {}
    for policy, rewards in results.items():
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        summary[policy] = (mean_r, std_r)
        print(f"{policy:12s}  mean={mean_r:8.1f}   std={std_r:7.1f}   "
              f"min={min(rewards):.0f}   max={max(rewards):.0f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    policies = list(results.keys())
    means = [summary[p][0] for p in policies]
    stds = [summary[p][1] for p in policies]
    colors = ["crimson", "gray", "steelblue"]

    ax.bar(policies, means, yerr=stds, capsize=8, color=colors, alpha=0.85)
    ax.set_ylabel(f"Cumulative Reward (mean +/- std over {NUM_TRIALS} trials)")
    ax.set_title(f"Policy Comparison: {EPISODES_PER_TRIAL} episodes x {NUM_TRIALS} trials")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")

    return summary


if __name__ == "__main__":
    results = run_multi_trial_evaluation()
    summarize_and_plot(results)
