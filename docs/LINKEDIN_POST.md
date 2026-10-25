# LinkedIn post draft

I just completed Task 1 of my AI project series: designing a practical AI problem before building anything.

Use case: routing IT helpdesk tickets into 5 queues (wifi, account, printing, software, hardware) with a "needs human" fallback when the model is unsure.

What I documented:
- User: the helpdesk coordinator who routes 100-300 tickets a day
- Data: 200 synthetic tickets + 20 hand-written challenge tickets
- Constraints: tiny data, CPU only, offline, explainable, abstains when unsure
- Evaluation: 5-fold CV, a held-out challenge set, two baselines, confusion matrix, error analysis

The honest finding: TF-IDF + logistic regression scored 1.00 macro-F1 in cross-validation but only 0.68 on my hand-written tickets, and a simple keyword baseline (0.85) actually beat it. Synthetic data made the problem look easier than it is, which is exactly why a separate challenge set matters.

Full write-up, code and results: <PASTE GITHUB REPO LINK>

#AI #MachineLearning #NLP #DataScience #Evaluation

---

# Video script (60 to 90 seconds, screen recording)

1. (0:00) "Hi, I'm <name>. This is my AI problem design for IT helpdesk ticket triage."
2. (0:10) Show docs/PROBLEM_DESIGN.md: "The user is a helpdesk coordinator. The task is classifying tickets into five queues, with a human fallback."
3. (0:25) Show data.py: "Data is 200 synthetic tickets plus 20 hand-written challenge tickets that are never trained on."
4. (0:40) Run `python train_eval.py` in the terminal.
5. (0:55) Point at results: "CV is 1.00, but the challenge set is 0.68, and keyword rules get 0.85."
6. (1:10) "So the next step is real tickets and better hardware and wifi coverage. Thanks for watching."
