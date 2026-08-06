"""
Evaluation script for the chat-style project.
1. Q-table heatmap: visualizes learned action preferences per state.
2. Baseline comparison: trained RL agent vs a naive "always Hint" policy.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, ACTIONS, STATE_SPACE, compute_reward, bucket_probability

DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFFICULTY_BIAS = {"Easy": 0.75, "Medium": 0.55, "Hard": 0.35}


def plot_q_table_heatmap(agent, save_path="plots/q_table_heatmap.png"):
    state_labels = [f"{p}\n{c}" for p, c in STATE_SPACE]
    q_matrix = np.array([agent.q_table[state] for state in STATE_SPACE])

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(q_matrix, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS)
    ax.set_yticks(range(len(state_labels)))
    ax.set_yticklabels(state_labels)

    for i in range(q_matrix.shape[0]):
        for j in range(q_matrix.shape[1]):
            ax.text(j, i, f"{q_matrix[i, j]:.1f}", ha="center", va="center",
                     color="black", fontsize=8)

    ax.set_title("Learned Q-values: (P(Correct) bucket, Calibration State) x Action")
    fig.colorbar(im, ax=ax, label="Q-value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def simulate_step(bn_model, hmm_model, agent, learner_history, confidence_level, previous_accuracy, fixed_action=None):
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
    action = fixed_action if fixed_action else agent.choose_action(rl_state)

    outcome = "needed_solution" if action == "Reveal" else (
        "improved_after_hint" if is_correct else "repeated_mistake"
    )
    reward = compute_reward(outcome)

    if fixed_action is None:
        next_state = (bucket_probability(0.5), calibration_state)
        agent.update(rl_state, action, reward, next_state)

    return updated_history, is_correct, reward


def run_policy(bn_model, hmm_model, agent, num_episodes, fixed_action=None):
    learner_history = []
    previous_accuracy = "Medium"
    cumulative_rewards = []
    total = 0

    for _ in range(num_episodes):
        confidence_level = random.choice(["Low", "Medium", "High"])
        learner_history, is_correct, reward = simulate_step(
            bn_model, hmm_model, agent, learner_history, confidence_level, previous_accuracy, fixed_action
        )
        total += reward
        cumulative_rewards.append(total)
        previous_accuracy = "High" if is_correct else "Low"

    return cumulative_rewards


def plot_comparison(baseline_rewards, rl_rewards, save_path="plots/rl_vs_baseline.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_rewards, label="Baseline (always Hint)", color="gray", linewidth=2)
    plt.plot(rl_rewards, label="Adaptive RL Agent (trained)", color="crimson", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Adaptive RL Agent vs. Naive Baseline (Cumulative Reward)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()

    trained_agent = TutorRLAgent(epsilon=0.0)
    plot_q_table_heatmap(trained_agent)

    rl_rewards = run_policy(bn_model, hmm_model, trained_agent, num_episodes=500)
    baseline_rewards = run_policy(bn_model, hmm_model, trained_agent, num_episodes=500, fixed_action="Hint")

    plot_comparison(baseline_rewards, rl_rewards)

    print(f"\nFinal cumulative reward - Baseline: {baseline_rewards[-1]}")
    print(f"Final cumulative reward - RL Agent: {rl_rewards[-1]}")
