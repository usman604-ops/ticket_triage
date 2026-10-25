# IT Helpdesk Ticket Triage

A narrow text-classification project: route short IT helpdesk tickets into one of five queues
(`wifi`, `account`, `printing`, `software`, `hardware`). Tickets the model is unsure about
(confidence below 0.50) are sent to `needs_human`.

## Use case summary

| Item | Description |
|---|---|
| User | Helpdesk coordinator routing 100 to 300 tickets per day |
| Data | 200 synthetic tickets (generated in `data.py`) plus 20 hand-written challenge tickets |
| Constraints | Tiny dataset, CPU only, offline, no cost per prediction, explainable, abstains when unsure |
| Model | TF-IDF (1-2 grams) + logistic regression |
| Evaluation | 5-fold CV, held-out challenge set, majority and keyword baselines, confusion matrix, error analysis, coverage vs selective accuracy |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_eval.py
python predict.py "Printer says offline" "Cannot reset my password" "asdf qwerty"
```

## Files

- `data.py` - synthetic data generator and the hand-written challenge set
- `train_eval.py` - trains, evaluates, saves `model.joblib` and `results.json`
- `predict.py` - routes new tickets, abstains below the threshold

## Results

Full write-up: [docs/PROBLEM_DESIGN.md](docs/PROBLEM_DESIGN.md)

| Setting | Majority | Keyword rules | TF-IDF + LogReg |
|---|---|---|---|
| CV macro-F1 | 0.067 | 0.795 | 1.000 |
| Challenge macro-F1 | 0.067 | 0.853 | 0.681 |
| Challenge accuracy | 0.200 | 0.850 | 0.700 |

Abstention at 0.50: coverage 0.30, selective accuracy 1.00.
The model does not yet meet the 0.80 challenge-set bar; the keyword baseline beats it.

## Limitations

Training data is synthetic, so cross-validation scores are optimistic. The challenge set is the more
honest number. Replace `build_train()` with real, anonymized tickets (`text,label`) before any real use.

## Acceptance bar

Macro-F1 of at least 0.80 on the challenge set, and at least 0.90 accuracy on tickets the model answers.
