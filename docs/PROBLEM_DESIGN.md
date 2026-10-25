# Task 1: AI Problem Design - IT Helpdesk Ticket Triage

## 1. Problem
Classify short IT helpdesk tickets (1 to 3 sentences) into one of five queues:
`wifi`, `account`, `printing`, `software`, `hardware`. If the model is not confident
(top probability < 0.50), the ticket goes to `needs_human`.

Narrow by design: one input type, one output label plus confidence, one decision (which queue).

## 2. User
| Item | Description |
|---|---|
| Primary user | Helpdesk coordinator at a small university or company IT desk, routing 100 to 300 tickets/day |
| Secondary user | Support technicians who receive already-sorted tickets |
| Pain point | Manual routing is repetitive and slow, and misrouted tickets bounce between queues |
| Success | Fewer misroutes, faster first response, clear visibility of uncertain tickets |
| Model output is seen by | Internal staff only, never end users |

## 3. Data source
| Item | Description |
|---|---|
| Training data | 200 synthetic tickets (40 per class) generated in `data.py` from templates with random openers, filler and cross-category distractors |
| Challenge set | 20 hand-written tickets in `data.py` (`CHALLENGE`), never used for training |
| Why synthetic | No private data, fully reproducible, no licensing issues |
| Real-world swap | Export `text,label` from the ticketing system, anonymize names/emails/IDs, replace `build_train()` |
| Limitation | Synthetic data is cleaner than real tickets, so cross-validation is optimistic. The challenge set is the honest number |

## 4. Constraints
| Constraint | Decision |
|---|---|
| Data size | Hundreds of rows, so a linear model on TF-IDF is appropriate |
| Latency | Under 50 ms per ticket, CPU only |
| Cost | Zero per prediction, runs offline with scikit-learn |
| Privacy | Ticket text never leaves the machine; real data must be anonymized |
| Interpretability | Top weighted words per class can be shown to coordinators |
| Error cost | A confident wrong route wastes technician time, so the system abstains below 0.50 |
| Scope | Five fixed labels; out-of-scope or non-English tickets should land in `needs_human` |
| Human in the loop | Staff can always override; overrides are logged for retraining |

## 5. Approach
1. Baseline A: majority-class predictor
2. Baseline B: keyword-rule classifier
3. Model: TF-IDF (1-2 grams) + logistic regression
4. Abstention: confidence < 0.50 becomes `needs_human`

## 6. Evaluation approach
| Question | Method | Metric |
|---|---|---|
| Generalizes within the data? | Stratified 5-fold CV on 200 synthetic tickets | Accuracy, macro-F1 |
| Survives realistic phrasing? | Train on 200 synthetic, test on 20 hand-written | Accuracy, macro-F1 |
| Better than simple options? | Compare with both baselines | Macro-F1 difference |
| Which classes are confused? | Confusion matrix, per-class report | Precision, recall |
| Is abstention useful? | Coverage and accuracy on answered tickets | Coverage, selective accuracy |
| What goes wrong? | Manual review of every challenge-set error | Qualitative |

Primary metric: macro-F1 on the challenge set.
Acceptance bar: macro-F1 >= 0.80 on the challenge set and >= 0.90 accuracy on answered tickets.

## 7. Results (from `python train_eval.py`, seed 42)
| Setting | Majority | Keyword rules | TF-IDF + LogReg |
|---|---|---|---|
| Cross-validation macro-F1 | 0.067 | 0.795 | 1.000 |
| Challenge-set macro-F1 | 0.067 | 0.853 | 0.681 |
| Challenge-set accuracy | 0.200 | 0.850 | 0.700 |

Abstention at 0.50: coverage 0.30, selective accuracy 1.00.

### Findings
- The model scores 1.00 in cross-validation but only 0.68 on hand-written tickets. The gap shows the synthetic data is too easy and the model partly learned template wording.
- The keyword baseline (0.85) beats the model on the challenge set, so the model does **not** meet the 0.80 acceptance bar yet.
- Errors cluster in `hardware` (1 of 4 correct) and `wifi` (2 of 4): e.g. "Screen flickers and then goes black" was predicted `account`, and "Can't get on the guest network" was predicted `account`.
- Abstention works: every ticket the model answered at 0.50 was correct, but it answers only 30 percent of tickets, so most would go to a human.

## 8. Next steps
1. Replace synthetic data with 500+ anonymized real tickets and re-run the same evaluation.
2. Add more varied phrasing for hardware and wifi, or combine keyword features with the model.
3. Tune the abstention threshold on a validation split.
4. Log technician overrides and retrain monthly.
5. Optionally compare with a zero-shot LLM classifier on the same challenge set (accuracy vs cost, latency, privacy).
