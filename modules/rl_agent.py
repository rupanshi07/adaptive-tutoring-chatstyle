"""
Module 5: Reinforcement Learning Agent
Learns the optimal tutoring action given (P(Correct) bucket, Calibration state).
"""

import numpy as np
import random
import json
import os

ACTIONS = ["Hint", "Explanation", "Retry", "Reveal"]
PROB_BUCKETS = ["Low", "Medium", "High"]       # P(Correct) discretized
CALIB_STATES = ["Over-confident", "Well-calibrated", "Under-confident"]

STATE_SPACE = [(p, c) for p in PROB_BUCKETS for c in CALIB_STATES]


def bucket_probability(p_correct):
    if p_correct < 0.34:
        return "Low"
    elif p_correct < 0.67:
        return "Medium"
    else:
        return "High"


class TutorRLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, q_table_path="data/q_table.json"):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.q_table = self._load_or_init_q_table()

    def _load_or_init_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "r") as f:
                raw = json.load(f)
            return {eval(k): np.array(v) for k, v in raw.items()}
        return {state: np.zeros(len(ACTIONS)) for state in STATE_SPACE}

    def save_q_table(self):
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        serializable = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(self.q_table_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        action_idx = ACTIONS.index(action)
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q


def compute_reward(outcome):
    """
    outcome: one of 'improved_after_hint', 'needed_solution', 'quit', 'repeated_mistake'
    """
    reward_map = {
        "improved_after_hint": 10,
        "needed_solution": 2,
        "quit": -10,
        "repeated_mistake": -5,
    }
    return reward_map.get(outcome, 0)


if __name__ == "__main__":
    agent = TutorRLAgent()

    # Simulate one interaction
    state = ("Low", "Over-confident")
    action = agent.choose_action(state)
    print(f"State: {state} -> Chosen action: {action}")

    # Simulate outcome and reward
    reward = compute_reward("improved_after_hint")
    next_state = ("Medium", "Well-calibrated")
    agent.update(state, action, reward, next_state)

    print(f"Q-value for {state}: {agent.q_table[state]}")
    agent.save_q_table()
    print("Q-table saved to data/q_table.json")
