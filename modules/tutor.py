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
        _client = genai.Client(api_key=api_key, http_options={"timeout": 30000})
    return _client


def generate_question(subject, topic, question_type, difficulty="Medium", max_retries=3):
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

    text, provider = call_llm_with_fallback(prompt)
    raw = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    parsed["question_type"] = question_type
    parsed["difficulty"] = difficulty
    parsed["_llm_provider"] = provider
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
        text, provider = call_llm_with_fallback(prompt)
        raw = text.replace("```json", "").replace("```", "").strip()
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
        prompt = _build_feedback_prompt(question, action)
        text, provider = call_llm_with_fallback(prompt)
        return text
    except Exception as e:
        return "[Tutor unavailable, fallback message] Action was: " + action + ". Error: " + str(e)


if __name__ == "__main__":
    q = generate_question("Computer Science", "Binary Search Trees", "MCQ", "Medium")
    print(json.dumps(q, indent=2))

    is_correct = grade_answer(q, q["correct_option"])
    print("Graded (using correct option):", is_correct)

    print("\nHint feedback:")
    print(generate_feedback(q, "Hint"))




_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _groq_client = Groq(api_key=api_key, timeout=20.0)
    return _groq_client


def call_llm_with_fallback(prompt, max_gemini_retries=1):
    """
    Tries Gemini first (with its own short retry loop for transient errors).
    If Gemini fails entirely, falls back to Groq (Llama 3.3 70B), a
    completely separate provider/infrastructure, so a Gemini outage does
    not take down question generation, grading, or feedback.
    Returns (response_text, provider_used) so callers can log which
    provider actually served the request.
    """
    import time as _time

    gemini_error = None
    try:
        client = get_client()
        for attempt in range(max_gemini_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=prompt,
                )
                return response.text.strip(), "gemini"
            except Exception as e:
                gemini_error = e
                _time.sleep(2 * (attempt + 1))
    except Exception as e:
        gemini_error = e

    try:
        groq_client = get_groq_client()
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip(), "groq"
    except Exception as groq_error:
        raise RuntimeError(
            f"Both Gemini and Groq failed. Gemini error: {gemini_error}. "
            f"Groq error: {groq_error}"
        )





CONF_TO_NUM = {"Low": 0, "Medium": 1, "High": 2}
NUM_TO_CONF = {0: "Low", 1: "Medium", 2: "High"}


def analyze_justification_confidence(question, justification):
    """
    Analyzes optional free-text justification for hedging vs assertive
    language, returning Low/Medium/High linguistic confidence, or None
    if no justification was given or analysis fails.
    """
    if not justification or not justification.strip():
        return None

    prompt = (
        "Question: " + question["question"] + "\n"
        "Student justification for their answer: " + justification + "\n\n"
        "Judge the STUDENT'S LINGUISTIC CONFIDENCE based only on their wording, "
        "not whether their reasoning is correct. Hedging language (maybe, "
        "I think, not sure, probably) suggests lower confidence. Assertive "
        "language (definitely, clearly, because, therefore) suggests higher "
        "confidence. "
        "Respond with ONLY valid JSON, no other text, in this exact format: "
        "{\"linguistic_confidence\": \"Low\"} or {\"linguistic_confidence\": \"Medium\"} or {\"linguistic_confidence\": \"High\"}"
    )

    try:
        text, provider = call_llm_with_fallback(prompt)
        raw = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        value = parsed.get("linguistic_confidence")
        if value in ("Low", "Medium", "High"):
            return value
        return None
    except Exception:
        return None


def blend_confidence(self_reported, linguistic):
    """
    Blends self-reported confidence with linguistic confidence derived
    from justification text. If no justification was given, returns the
    self-reported value unchanged.
    """
    if linguistic is None:
        return self_reported
    avg = (CONF_TO_NUM[self_reported] + CONF_TO_NUM[linguistic]) / 2
    return NUM_TO_CONF[round(avg)]





