"""Route a ticket: python predict.py "my printer is jammed" "wifi is down" """
import sys
import joblib

THRESHOLD = 0.50


def route(model, text):
    proba = model.predict_proba([text])[0]
    i = proba.argmax()
    label, conf = model.classes_[i], float(proba[i])
    if conf < THRESHOLD:
        return "needs_human", conf
    return label, conf


if __name__ == "__main__":
    model = joblib.load("model.joblib")
    for text in sys.argv[1:]:
        label, conf = route(model, text)
        print(f"{label:12s} ({conf:.2f})  <- {text}")
