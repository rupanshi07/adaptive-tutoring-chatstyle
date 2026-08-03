"""
Module 1: Conversational Intelligent Tutor (Gemini-powered)
Generates questions dynamically based on Subject, Topic, and Question Type,
then grades answers and produces adaptive feedback.
"""

import os
import json
from google import genai

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_question(subject, topic, question_type, difficulty="Medium"):
    """
    question_type: 'MCQ', 'Descriptive', or 'Coding'
    Returns a dict with: question, question_type, difficulty,
    correct_answer_summary, and (for MCQ only) options + correct_option.
    """
    if question_type == "MCQ":
        format_instruction = (
            "Provide the question, exactly 4 answer options labeled A-D, "
            "and which single letter is correct. "
            "Respond with ONLY valid JSON in this exact format: "
            "{\"question\": \"...\", \"options\": {\"A\": \"...\", \"B\": \"...\", "
            "\"C\": \"...\", \"D\": \"...\"}, \"correct_option\": \"A\", "
            "\"correct_answer_summary\": \"...\"}"
        )
    elif question_type == "Coding":
        format_instruction = (
            "Provide a coding problem statement (no starter code needed) and "
            "a short summary of what a correct solution should do or return. "
            "Respond with ONLY valid JSON in this exact format: "
            "{\"question\": \"...\", \"correct_answer_summary\": \"...\"}"
        )
    else:
        format_instruction = (
            "Provide a descriptive/theory question and a concise reference "
            "answer summary. "
            "Respond with ONLY valid JSON in this exact format: "
            "{\"question\": \"...\", \"correct_answer_summary\": \"...\"}"
        )

    prompt = (
        "Subject: " + subject + "\n"
        "Topic: " + topic + "\n"
        "Difficulty: " + difficulty + "\n\n"
        "Generate one " + question_type + " practice question for a student "
        "studying this subject and topic. " + format_instruction
    )

    client = get_client()
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    parsed["question_type"] = question_type
    parsed["difficulty"] = difficulty
    return parsed


def grade_answer(question, student_answer):
    """
    For MCQ: compares selected option letter directly (no LLM call needed).
    For Descriptive/Coding: uses Gemini to judge correctness.
    Returns True/False.
    """
    if question["question_type"] == "MCQ":
        return student_answer.strip().upper() == question["correct_option"].strip().upper()

    prompt = (
        "Question: " + question["question"] + "\n"
        "Reference correct answer: " + question["correct_answer_summary"] + "\n"
        "Student answer: " + student_answer + "\n\n"
        "Judge if the student answer is substantively correct, even if worded "
        "differently or less formally than the reference answer. "
        "Respond with ONLY valid JSON, no other text, in this exact format: "
        "{\"correct\": true} or {\"correct\": false}"
    )

    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return bool(parsed.get("correct", False))
    except Exception:
        return False


def _build_feedback_prompt(question, action):
    base = (
        "You are an adaptive tutor helping a student with this question:\n"
        "\"" + question["question"] + "\"\n"
        "The correct answer is: " + question["correct_answer_summary"] + "\n\n"
    )
    if action == "Hint":
        return base + (
            "Give ONE short hint (max 2 sentences) that nudges the student "
            "toward the answer WITHOUT revealing it directly."
        )
    elif action == "Explanation":
        return base + (
            "Give a clear, concise explanation (3-4 sentences) of the correct "
            "answer, written for a student who just got this wrong or wants "
            "more depth."
        )
    elif action == "Retry":
        return base + (
            "Write ONE short, encouraging sentence telling the student to try "
            "again without giving any hint or revealing the answer."
        )
    elif action == "Reveal":
        return base + (
            "Clearly state the correct answer in 1-2 sentences, in a "
            "supportive tone."
        )
    else:
        return base + "Give a brief, friendly transition to the next question."


def generate_feedback(question, action, use_llm=True):
    if not use_llm:
        return "[static] Action=" + action
    try:
        client = get_client()
        prompt = _build_feedback_prompt(question, action)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return "[Tutor unavailable, fallback message] Action was: " + action + ". Error: " + str(e)


if __name__ == "__main__":
    q = generate_question("Computer Science", "Binary Search Trees", "MCQ", "Medium")
    print(json.dumps(q, indent=2))

    is_correct = grade_answer(q, q["correct_option"])
    print("Graded (using correct option):", is_correct)

    print("\nHint feedback:")
    print(generate_feedback(q, "Hint"))
