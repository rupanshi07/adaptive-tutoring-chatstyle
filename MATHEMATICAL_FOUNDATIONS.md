\# Mathematical Foundations of the Adaptive Tutoring System



This document outlines the probabilistic models and statistical architectures implemented in the tutoring system.



\## 1. Bayesian Networks (BN) - modules/bayesian\_network.py

We utilize a Directed Acyclic Graph (DAG) to model the conditional dependencies between student interactions and their likelihood of answering correctly.

\* \*\*Nodes:\*\* Confidence Level, Question Difficulty, Response Time, Number of Hints, Previous Accuracy, and Correct Answer.

\* \*\*Inference:\*\* Using the pgmpy library, we calculate the posterior probability P(Correct | Evidence) using Bayesian Inference. This allows the system to distinguish between a student who guessed correctly and a student who actually understands the material.



\## 2. Hidden Markov Models (HMM) - modules/hmm\_calibration.py

A student's true "knowledge state" is hidden; we can only observe their interactions (answers, time taken, hints).

\* \*\*Hidden States:\*\* e.g., Unconfident, Calibrated, Overconfident.

\* \*\*Observable Emissions:\*\* Response time, hint requests, and correctness.

\* We use the hmmlearn library to perform the Viterbi algorithm, estimating the most likely sequence of hidden knowledge states the student is currently in.



\## 3. Real-World Parameter Learning - dataset\_bayesian\_network.py

Instead of hardcoding probabilities, we trained our Bayesian Network's Conditional Probability Distributions (CPDs) using Maximum Likelihood Estimation (MLE) on the real-world UCI Student Performance Dataset.



\## 4. Markov Decision Process (Q-Learning) - modules/rl\_agent.py

The system acts as a Reinforcement Learning agent trying to maximize the student's learning (Reward).

\* \*\*State:\*\* The probability estimates from the BN and HMM.

\* \*\*Action:\*\* Give a Hint, Give Explanation, Ask to Retry, or Reveal Answer.

\* \*\*Update Rule:\*\* We use the Bellman Equation to update our Q-table dynamically based on student feedback.

