"""
Module 6: ChatGPT-style conversational UI with RLHF satisfaction feedback.
Confidence is chosen via a sidebar panel; hints are requested via a button.
Chat is used for Subject/Topic/Type selection, answers, and next-question flow.
"""

import streamlit as st
import time
import csv
import os

from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, bucket_probability, compute_reward
from modules.tutor import generate_question, grade_answer, generate_feedback

LOG_PATH = "data/interaction_log.csv"
LOG_FIELDS = [
    "timestamp", "subject", "topic", "question_type", "difficulty",
    "confidence", "response_time", "elapsed_seconds", "correct",
    "calibration_state", "p_correct", "action", "base_reward",
    "satisfaction", "combined_reward", "previous_accuracy", "hint_used",
]


def log_interaction(row):
    file_exists = os.path.exists(LOG_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@st.cache_resource
def load_models():
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()
    agent = TutorRLAgent(epsilon=0.1)
    return bn_model, hmm_model, agent


def init_session_state():
    defaults = {
        "stage": "ask_subject",
        "subject": None,
        "topic": None,
        "question_type": None,
        "current_question": None,
        "question_start_time": None,
        "history": [],
        "previous_accuracy": "Medium",
        "round_num": 1,
        "chat_log": [],
        "pending_update": None,
        "retry_same_question": False,
        "hint_used_this_round": False,
        "hint_text": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_message(role, content):
    st.session_state.chat_log.append({"role": role, "content": content})


def render_chat_log():
    for msg in st.session_state.chat_log:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def start_new_question():
    with st.spinner("Generating your question..."):
        try:
            q = generate_question(
                st.session_state.subject,
                st.session_state.topic,
                st.session_state.question_type,
            )
        except Exception:
            add_message("assistant", "Sorry, the question generator is temporarily unavailable. Please type 'next' to try again in a moment.")
            st.session_state.stage = "await_next"
            return

    st.session_state.current_question = q
    st.session_state.question_start_time = time.time()
    st.session_state.hint_used_this_round = False
    st.session_state.hint_text = None
    st.session_state.stage = "quiz_answer"

    if q["question_type"] == "MCQ":
        options_text = "\n".join(f"**{k}.** {v}" for k, v in q["options"].items())
        add_message("assistant", f"**Round {st.session_state.round_num} ({q['difficulty']})**\n\n{q['question']}\n\n{options_text}")
    else:
        add_message("assistant", f"**Round {st.session_state.round_num} ({q['difficulty']})**\n\n{q['question']}")


def process_answer(bn_model, hmm_model, agent, student_answer, confidence_level):
    q = st.session_state.current_question

    elapsed = time.time() - st.session_state.question_start_time
    time_taken = "Fast" if elapsed < 15 else "Slow"
    hints_flag = "Yes" if st.session_state.hint_used_this_round else "No"

    # Bayesian Network predicts BEFORE grading -- genuine prediction,
    # not a redundant re-derivation of an already-known outcome.
    p_correct = estimate_probability_correct(
        bn_model, confidence=confidence_level, difficulty=q["difficulty"],
        time_=time_taken, hints=hints_flag, previous_accuracy=st.session_state.previous_accuracy,
    )

    with st.spinner("Grading..."):
        is_correct = grade_answer(q, student_answer)

    observation = encode_observation(confidence_level, is_correct)
    updated_history = st.session_state.history + [observation]
    if len(updated_history) > 20:
        updated_history = updated_history[-20:]
    calibration_state = infer_calibration_state(hmm_model, updated_history)

    prob_bucket = bucket_probability(p_correct)
    rl_state = (prob_bucket, calibration_state)
    action = agent.choose_action(rl_state)

    with st.spinner("Preparing feedback..."):
        feedback_text = generate_feedback(q, action)

    outcome = "needed_solution" if action == "Reveal" else (
        "improved_after_hint" if is_correct else "repeated_mistake"
    )
    base_reward = compute_reward(outcome)

    st.session_state.history = updated_history
    st.session_state.previous_accuracy = "High" if is_correct else "Low"
    st.session_state.retry_same_question = (action == "Retry" and not is_correct)

    verdict = "Correct" if is_correct else "Incorrect"
    reply = (
        f"**Your answer:** {student_answer}\n\n"
        f"**Graded as:** {verdict}\n\n"
        f"*P(Correct) estimate: {p_correct:.0%} | Calibration: {calibration_state} | "
        f"Action: {action} | Response time: {time_taken} ({elapsed:.1f}s) | Hint used: {hints_flag}*\n\n"
        f"{feedback_text}"
    )
    add_message("assistant", reply)

    st.session_state.pending_update = {
        "rl_state": rl_state,
        "action": action,
        "calibration_state": calibration_state,
        "base_reward": base_reward,
        "p_correct": p_correct,
        "is_correct": is_correct,
        "confidence": confidence_level,
        "response_time": time_taken,
        "elapsed_seconds": round(elapsed, 1),
        "hint_used": hints_flag,
    }


def finalize_with_satisfaction(agent, satisfied):
    pu = st.session_state.pending_update
    satisfaction_bonus = 5 if satisfied else -5
    combined_reward = pu["base_reward"] + satisfaction_bonus

    next_state = (bucket_probability(0.5), pu["calibration_state"])
    agent.update(pu["rl_state"], pu["action"], combined_reward, next_state)
    agent.save_q_table()

    log_interaction({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subject": st.session_state.subject,
        "topic": st.session_state.topic,
        "question_type": st.session_state.question_type,
        "difficulty": st.session_state.current_question["difficulty"],
        "confidence": pu["confidence"],
        "response_time": pu["response_time"],
        "elapsed_seconds": pu["elapsed_seconds"],
        "correct": pu["is_correct"],
        "calibration_state": pu["calibration_state"],
        "p_correct": round(pu["p_correct"], 3),
        "action": pu["action"],
        "base_reward": pu["base_reward"],
        "satisfaction": "Helpful" if satisfied else "Not Helpful",
        "combined_reward": combined_reward,
        "previous_accuracy": st.session_state.previous_accuracy,
        "hint_used": pu["hint_used"],
    })

    if st.session_state.retry_same_question:
        add_message("assistant", "No problem -- give this same question another shot below.")
        st.session_state.question_start_time = time.time()
        st.session_state.stage = "quiz_answer"
    else:
        add_message("assistant", "Thanks for the feedback! Type 'next' for another question.")
        st.session_state.stage = "await_next"

    st.session_state.pending_update = None


def main():
    st.set_page_config(page_title="Adaptive Tutor Chat", page_icon=":books:")
    st.title("Adaptive Tutoring Chat")
    st.caption("Bayesian Networks + Hidden Markov Models + Reinforcement Learning from Human Feedback")

    bn_model, hmm_model, agent = load_models()
    init_session_state()

    with st.sidebar:
        st.subheader("Your Confidence")
        confidence_level = st.radio(
            "How confident are you in your next answer?",
            ["Low", "Medium", "High"],
            index=1,
            key="confidence_choice",
        )
        st.divider()
        st.subheader("Session Info")
        if st.session_state.subject:
            st.write(f"**Subject:** {st.session_state.subject}")
        if st.session_state.topic:
            st.write(f"**Topic:** {st.session_state.topic}")
        if st.session_state.question_type:
            st.write(f"**Type:** {st.session_state.question_type}")
        st.write(f"**Previous Accuracy:** {st.session_state.previous_accuracy}")
        st.write(f"**History length:** {len(st.session_state.history)}")
        st.caption("All interactions logged to data/interaction_log.csv")

        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if not st.session_state.chat_log:
        add_message("assistant", "Hi! What subject would you like to practice today? (e.g. Computer Science, Mathematics, Physics)")

    render_chat_log()

    # Hint button: only shown while actively answering a question, before satisfaction prompt
    if st.session_state.stage == "quiz_answer" and st.session_state.current_question and st.session_state.pending_update is None:
        if not st.session_state.hint_used_this_round:
            if st.button("Get Hint"):
                st.session_state.hint_used_this_round = True
                with st.spinner("Generating hint..."):
                    hint_text = generate_feedback(st.session_state.current_question, "Hint")
                st.session_state.hint_text = hint_text
                add_message("assistant", hint_text)
                st.rerun()
        elif st.session_state.hint_text:
            st.caption("Hint already used for this question.")

    # Satisfaction prompt takes priority over chat input
    if st.session_state.pending_update is not None:
        st.divider()
        st.markdown("**Was that feedback helpful?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Helpful", use_container_width=True):
                finalize_with_satisfaction(agent, satisfied=True)
                st.rerun()
        with col2:
            if st.button("Not Helpful", use_container_width=True):
                finalize_with_satisfaction(agent, satisfied=False)
                st.rerun()
        return

    user_input = st.chat_input("Type your response...")

    if user_input:
        add_message("user", user_input)

        if st.session_state.stage == "ask_subject":
            st.session_state.subject = user_input
            st.session_state.stage = "ask_topic"
            add_message("assistant", f"Great, {user_input}! What topic within that subject? (e.g. Binary Search Trees, Thermodynamics, Calculus)")

        elif st.session_state.stage == "ask_topic":
            st.session_state.topic = user_input
            st.session_state.stage = "ask_type"
            add_message("assistant", "What kind of question would you like? Type one of: MCQ, Descriptive, or Coding")

        elif st.session_state.stage == "ask_type":
            normalized = user_input.strip().lower()
            if normalized in ["mcq", "multiple choice"]:
                st.session_state.question_type = "MCQ"
            elif normalized in ["coding", "code"]:
                st.session_state.question_type = "Coding"
            else:
                st.session_state.question_type = "Descriptive"
            add_message("assistant", "Set your confidence in the sidebar, then answer below once the question appears.")
            start_new_question()

        elif st.session_state.stage == "quiz_answer":
            process_answer(bn_model, hmm_model, agent, user_input, confidence_level)
            st.session_state.round_num += 1

        elif st.session_state.stage == "await_next":
            if user_input.strip().lower() in ["yes", "next", "continue"]:
                start_new_question()
            else:
                add_message("assistant", "Type 'next' when you would like another question.")

        st.rerun()


if __name__ == "__main__":
    main()

