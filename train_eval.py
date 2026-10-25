"""Train, evaluate, and save the ticket classifier."""
import json
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from data import CHALLENGE, LABELS, build_train

THRESHOLD = 0.50
SEED = 42

KEYWORDS = {
    "wifi": ["wifi", "wi-fi", "network", "internet", "eduroam", "connection", "signal"],
    "account": ["password", "account", "login", "log in", "credentials", "authentication"],
    "printing": ["print", "printer", "toner", "copier", "paper"],
    "software": ["install", "license", "software", "crash", "update", "app"],
    "hardware": ["laptop", "screen", "monitor", "keyboard", "battery", "charger"],
}


class KeywordBaseline:
    """Pick the label with the most keyword hits; ties go to the first label."""
    def predict(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            scores = {l: sum(k in t for k in kws) for l, kws in KEYWORDS.items()}
            out.append(max(LABELS, key=lambda l: scores[l]))
        return out


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=5.0),
    )


def report(name, y_true, y_pred):
    return {
        "name": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
    }


def main():
    rows = build_train()
    X = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    Xc = [r[0] for r in CHALLENGE]
    yc = np.array([r[1] for r in CHALLENGE])

    results = {}

    # 1) Cross-validation on synthetic data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_pred = cross_val_predict(make_model(), X, y, cv=cv)
    dummy_pred = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=cv)
    kw_pred = KeywordBaseline().predict(X)
    results["cross_validation"] = [
        report("majority baseline", y, dummy_pred),
        report("keyword baseline", y, kw_pred),
        report("tfidf+logreg", y, cv_pred),
    ]

    # 2) Fit on all synthetic data, test on hand-written challenge set
    model = make_model().fit(X, y)
    proba = model.predict_proba(Xc)
    pred = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)

    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    results["challenge_set"] = [
        report("majority baseline", yc, dummy.predict(Xc)),
        report("keyword baseline", yc, KeywordBaseline().predict(Xc)),
        report("tfidf+logreg", yc, pred),
    ]

    # 3) Abstention analysis
    answered = conf >= THRESHOLD
    coverage = float(answered.mean())
    sel_acc = float((pred[answered] == yc[answered]).mean()) if answered.any() else float("nan")
    results["abstention"] = {
        "threshold": THRESHOLD,
        "coverage": round(coverage, 3),
        "selective_accuracy": round(sel_acc, 3),
    }

    # 4) Detail printouts
    print("=== Cross-validation (synthetic, 5-fold) ===")
    for r in results["cross_validation"]:
        print(r)
    print("\n=== Challenge set (20 hand-written tickets) ===")
    for r in results["challenge_set"]:
        print(r)
    print("\nPer-class report (challenge set):")
    print(classification_report(yc, pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), label order:", LABELS)
    print(confusion_matrix(yc, pred, labels=LABELS))
    print(f"\nAbstention at {THRESHOLD}: coverage={coverage:.2f}, "
          f"selective accuracy={sel_acc:.2f}")

    print("\n=== Error analysis (challenge set) ===")
    for text, true, p, c in zip(Xc, yc, pred, conf):
        if true != p:
            print(f"- true={true:9s} pred={p:9s} conf={c:.2f} | {text}")

    # 5) Interpretability: top words per class
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    names = np.array(vec.get_feature_names_out())
    print("\n=== Top terms per class ===")
    for i, label in enumerate(clf.classes_):
        top = names[np.argsort(clf.coef_[i])[-6:][::-1]]
        print(f"{label:9s}: {', '.join(top)}")

    joblib.dump(model, "model.joblib")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved model.joblib and results.json")


if __name__ == "__main__":
    main()
