# \# Adaptive Tutoring System — Chat-Style Interface

# 

# A conversational adaptive tutoring system that dynamically generates questions based on Subject, Topic, and Question Type, then personalizes feedback using Bayesian Networks, Hidden Markov Models, and learner-feedback-driven Reinforcement Learning.

# 

# \## Overview

# 

# The student talks to the tutor like a chat assistant:

# 

# 1\. "What subject would you like to practice?"

# 2\. "What topic within that subject?"

# 3\. "MCQ, Descriptive, or Coding?"

# 

# Gemini (with Groq as an automatic fallback provider) generates a fresh question matching those choices. Before the student answers, the system shows a preliminary correctness estimate using only the question's difficulty and the student's history. Once they answer -- optionally including a short justification of their reasoning -- a Bayesian Network, a Hidden Markov Model, and a Reinforcement Learning agent jointly decide how to respond.

# 

# \## Architecture

# 

# ```

# Subject / Topic / Question Type (chat input)

# &#x20;       |

# &#x20;       v

# Gemini/Groq: generate\_question() --> dynamic question + reference answer

# &#x20;       |

# &#x20;       v

# Preliminary Bayesian Network query (Difficulty + PreviousAccuracy only)

# \--> shown to student BEFORE they answer, as a partial-evidence estimate

# &#x20;       |

# &#x20;       v

# Student sets Confidence (sidebar), optionally requests a Hint (button),

# optionally writes a justification for their answer

# &#x20;       |

# &#x20;       v

# IN PARALLEL: Gemini/Groq grades the answer (MCQ = exact string match,

# no LLM call) AND analyzes the justification text for hedging vs.

# assertive language --> linguistic confidence

# &#x20;       |

# &#x20;       v

# Self-reported confidence + linguistic confidence are blended into one

# effective confidence value

# &#x20;       |

# &#x20;       v

# Full Bayesian Network query (all 5 evidence variables now known:

# Confidence, Difficulty, Time, Hints, PreviousAccuracy) --> P(Correct)

# &#x20;       |

# &#x20;       v

# Hidden Markov Model (Viterbi decoding over interaction history)

# \--> Calibration state: Over-confident / Well-calibrated / Under-confident

# &#x20;       |

# &#x20;       v

# Reinforcement Learning Agent (tabular Q-learning)

# \--> Selected Action: Hint / Explanation / Retry / Reveal

# &#x20;       |

# &#x20;       v

# Gemini/Groq: generate\_feedback() --> NL text shown to student

# &#x20;       |

# &#x20;       v

# Student rates: Helpful / Not Helpful

# &#x20;       |

# &#x20;       v

# combined\_reward = outcome-based reward + satisfaction bonus

# &#x20;       |

# &#x20;       v

# Q-table updated and persisted; interaction logged to CSV

# &#x20;       |

# &#x20;       v

# If action was Retry and the answer was wrong: same question repeats.

# Otherwise: next question.

# ```

# 

# \## The Bayesian Network: Structure

# 

# The live network is a genuine multi-layer DAG, not a naive/star structure:

# 

# ```

# PreviousAccuracy --------> Confidence

# Difficulty -------------> Hints

# Difficulty, Hints -------> Time

# Confidence, Difficulty, Time, Hints, PreviousAccuracy -> Correct

# ```

# 

# Two ways this network is queried:

# 

# \- \*\*Preliminary query\*\* (before the student answers): evidence = {Difficulty, PreviousAccuracy} only. Confidence, Hints, and Time are genuinely marginalized over, using the upstream edges above -- this is real inference under partial evidence.

# \- \*\*Full query\*\* (after the student answers): evidence = all five variables, fully observed. This collapses to a direct CPD lookup, same as a simpler network would give -- the richer structure specifically enables the preliminary query, which a naive/star structure could not support at all.

# 

# CPD values for the Correct node are still expert-specified (a hand-picked linear scoring formula), not learned from data. A separate, standalone network (`dataset\_bayesian\_network.py`) demonstrates real parameter learning from the real UCI Student Performance dataset (Cortez, 2008, 649 students) via pgmpy's BayesianEstimator -- but that network is not wired into the live app, since no public dataset captures live self-reported confidence or hint-usage signals.

# 

# \## The Hidden Markov Model

# 

# Three hidden states (Over-confident, Well-calibrated, Under-confident), inferred via Viterbi decoding over a sequence of (confidence, correctness) observations. The transition, emission, and initial-state matrices are currently hand-specified.

# 

# Two supporting pieces exist:

# 

# \- `hmm\_state\_definition.py` gives a formal, checkable, rule-based definition of the three states (a rolling mean calibration-error threshold), independent of the hand-picked matrices.

# \- `learn\_hmm\_from\_log.py` attempts real Baum-Welch parameter learning from `data/interaction\_log.csv`, but explicitly refuses to run unless at least 50 real logged interactions exist with at least 4 of 6 possible observation types represented -- this is intentional, to avoid producing an overfit or meaningless model on insufficient data.

# 

# \## Reinforcement Learning

# 

# Tabular Q-learning over 9 states (P(Correct) bucket x calibration state) and 4 actions (Hint, Explanation, Retry, Reveal). Trained offline on 3,000 simulated episodes (`train.py`). Reward combines an automatic outcome-based signal with a real learner satisfaction rating (Helpful / Not Helpful).

# 

# Evaluated rigorously across 10 independent trials against two baselines (`evaluate\_rl\_robust.py`): the trained policy clearly beats a random policy, but does not show a statistically convincing advantage over a simple fixed "always Hint" policy at the current sample size -- this is reported honestly rather than only showing favorable comparisons.

# 

# \## Terminology Note: "RLHF"

# 

# This project uses learner feedback (Helpful / Not Helpful ratings) as part of the reward signal for a tabular Q-learning tutoring policy. This is reinforcement learning that uses human/learner feedback -- but it is NOT the same as conventional NLP RLHF (the technique used to align large language models), which involves training a separate reward model on human preference data and using it to fine-tune the LLM's own weights via policy-gradient methods.

# 

# In this project, Gemini and Groq are never fine-tuned; their weights are never modified. Only the separate, from-scratch Q-learning agent's policy is updated using learner feedback. The more precise term for this is "learner-feedback-driven reinforcement learning" or "RLHF-inspired reinforcement learning."

# 

# \## LLM Integration and Resilience

# 

# Google Gemini (gemini-flash-latest) is the primary provider for question generation, free-text grading, and feedback text. Groq (openai/gpt-oss-120b) is an automatic fallback if Gemini fails or times out -- both clients have explicit timeouts (30s / 20s) to prevent multi-minute hangs if a request stalls rather than cleanly failing.

# 

# If grading itself fails (e.g., both providers are unreachable), the student is told grading failed and is NOT silently marked wrong -- this distinguishes an LLM/API failure from a genuinely incorrect answer.

# 

# \## Project Structure

# 

# ```

# Project/

# |-- modules/

# |   |-- bayesian\_network.py     # Live BN: structure, full query, preliminary query

# |   |-- hmm\_calibration.py      # HMM: hand-specified matrices, Viterbi inference

# |   |-- rl\_agent.py             # Tabular Q-learning tutoring agent

# |   |-- tutor.py                # Gemini/Groq: questions, grading, feedback, justification analysis

# |-- data/

# |   |-- q\_table.json            # Persisted, learned Q-table

# |   |-- interaction\_log.csv     # Real logged interactions (grows with real usage)

# |   |-- student\_performance\_raw.csv   # Generated by fetch\_dataset.py (not committed)

# |-- fetch\_dataset.py            # Downloads UCI Student Performance dataset

# |-- dataset\_bayesian\_network.py # Learns CPDs from real data using pgmpy (standalone demo)

# |-- hmm\_state\_definition.py     # Formal, rule-based calibration state definition

# |-- learn\_hmm\_from\_log.py       # Attempts to learn HMM parameters from real logs (gated)

# |-- train.py                    # Batch-trains the RL agent (simulated, no LLM calls)

# |-- evaluate.py                 # Single-run RL vs. fixed baseline + Q-table heatmap

# |-- evaluate\_rl\_robust.py       # 10-trial RL vs. random and fixed baselines

# |-- evaluate\_calibration.py     # Brier score + calibration curve on real logged data

# |-- app.py                      # The chat-style Streamlit application

# |-- requirements.txt

# ```

# 

# \## Setup

# 

# 1\. Clone the repository and enter the project folder:

# &#x20;  ```

# &#x20;  git clone https://github.com/rupanshi07/adaptive-tutoring-chatstyle.git

# &#x20;  cd adaptive-tutoring-chatstyle

# &#x20;  ```

# 

# 2\. Create and activate a virtual environment:

# &#x20;  ```

# &#x20;  python -m venv venv

# &#x20;  venv\\Scripts\\activate

# &#x20;  ```

# 

# 3\. Install dependencies:

# &#x20;  ```

# &#x20;  pip install -r requirements.txt

# &#x20;  ```

# 

# 4\. Get a free Gemini API key from https://aistudio.google.com/app/apikey and a free Groq API key from https://console.groq.com/keys (both have no-cost tiers, no credit card required), then set them as environment variables:

# &#x20;  ```

# &#x20;  setx GEMINI\_API\_KEY "your-gemini-key-here"

# &#x20;  setx GROQ\_API\_KEY "your-groq-key-here"

# &#x20;  ```

# &#x20;  Close and reopen your terminal after running this.

# 

# \## Running the Project

# 

# Run the chat-style demo:

# ```

# streamlit run app.py

# ```

# 

# Fetch the sourced dataset:

# ```

# python fetch\_dataset.py

# ```

# 

# Learn Bayesian Network parameters from real data:

# ```

# python dataset\_bayesian\_network.py

# ```

# 

# Attempt to learn HMM parameters from real logged interactions:

# ```

# python learn\_hmm\_from\_log.py

# ```

# 

# Train the RL agent (simulated, 3000 episodes):

# ```

# python train.py

# ```

# 

# Run rigorous multi-trial RL evaluation:

# ```

# python evaluate\_rl\_robust.py

# ```

# 

# Check Bayesian Network calibration on real logged data:

# ```

# python evaluate\_calibration.py

# ```

# 

# \## Dataset Citation

# 

# Cortez, P. (2008). Student Performance \[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

# 

# \## Tech Stack

# 

# \- pgmpy -- Bayesian Network construction, exact inference (Variable Elimination), and parameter learning

# \- hmmlearn -- Hidden Markov Model, Viterbi decoding, (gated) Baum-Welch learning

# \- Google Gemini API (google-genai) and Groq API (groq) -- dynamic question generation, grading, feedback, with automatic fallback

# \- ucimlrepo -- fetching the sourced UCI dataset

# \- Streamlit -- conversational chat-style web UI

# 

# \## Known Limitations (stated honestly, not hidden)

# 

# \- The live Bayesian Network's Correct-node CPD is still expert-specified, not learned from real data; a script to relearn it from `interaction\_log.csv` (analogous to `learn\_hmm\_from\_log.py`) does not exist yet.

# \- The HMM's actual matrices remain hand-specified; real learning is implemented but correctly gated behind a data-sufficiency check that is not yet met.

# \- Only a small number of real interactions have been logged so far -- the Brier score/calibration curve and any future data-driven learning are not yet based on a meaningful sample size.

# \- `train.py`'s simulated learner behavior uses an arbitrary difficulty-success heuristic, not grounded in the real UCI dataset's actual pass-rate patterns.

# \- The RL agent shows a clear advantage over random behavior but not a statistically convincing advantage over a simple fixed "always Hint" baseline, based on rigorous 10-trial testing.

# \- The dataset-driven Bayesian Network (UCI-based) has not itself been evaluated for calibration, and the source dataset's class imbalance (about 85% Pass / 15% Fail) was not addressed.



