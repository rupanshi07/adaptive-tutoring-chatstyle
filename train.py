"""
Batch training script for the chat-style project.
Trains the RL agent using simulated interactions (no Gemini calls needed --
the agent only needs difficulty/correctness patterns to learn from, not
real question text).
"""

import random
from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, bucket_probability, compute_reward

NUM_EPISODES = 3000
DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFFICULTY_BIAS = {"Easy": 0.75, "Medium": 0.55, "Hard": 0.35}


def simulate_episode(bn_model, hmm_model, agent, learner_history, confidence_level, previous_accuracy):
    difficulty = random.choice(DIFFICULTIES)
    is_correct = random.random() < DIFFICULTY_BIAS[difficulty]
    time_taken = random.choice(["Fast", "Slow"])

    p_correct = estimate_probability_correct(
        bn_model,
        confidence=confidence_level,
        difficulty=difficulty,
        time_=time_taken,
        hints="No",
        previous_accuracy=previous_accuracy,
    )

    observation = encode_observation(confidence_level, is_correct)
    updated_history = learner_history + [observation]
    if len(updated_history) > 20:
        updated_history = updated_history[-20:]
    calibration_state = infer_calibration_state(hmm_model, updated_history)

    prob_bucket = bucket_probability(p_correct)
    rl_state = (prob_bucket, calibration_state)
    action = agent.choose_action(rl_state)

    outcome = "needed_solution" if action == "Reveal" else (
        "improved_after_hint" if is_correct else "repeated_mistake"
    )
    reward = compute_reward(outcome)
    next_state = (bucket_probability(0.5), calibration_state)
    agent.update(rl_state, action, reward, next_state)

    return updated_history, is_correct


def train():
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()
    agent = TutorRLAgent(epsilon=0.3)

    learner_history = []
    previous_accuracy = "Medium"

    for i in range(NUM_EPISODES):
        confidence_level = random.choice(["Low", "Medium", "High"])
        learner_history, is_correct = simulate_episode(
            bn_model, hmm_model, agent, learner_history, confidence_level, previous_accuracy
        )
        previous_accuracy = "High" if is_correct else "Low"

        if (i + 1) % 500 == 0:
            print(f"Episode {i + 1}/{NUM_EPISODES} complete")

    agent.save_q_table()
    print("\nTraining complete. Final Q-table:")
    for state, q_values in agent.q_table.items():
        print(f"{state}: {q_values}")


if __name__ == "__main__":
    train()
