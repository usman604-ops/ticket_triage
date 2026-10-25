"""Synthetic training data plus a hand-written challenge set."""
import random

LABELS = ["account", "hardware", "printing", "software", "wifi"]

OPENERS = ["", "", "Hi,", "Hello team,", "Urgent:", "Good morning,", "Please help."]
FILLER = ["", "", "Thanks.", "Please advise.", "This is blocking my work.",
          "It started today.", "Happened twice this week."]

TEMPLATES = {
    "wifi": {
        "subject": ["the wifi", "campus wi-fi", "eduroam", "my internet connection",
                    "the network"],
        "problem": ["keeps disconnecting", "is extremely slow", "will not connect",
                    "says connected but no internet", "drops every few minutes"],
        "place": ["in the library", "in my dorm room", "in lecture hall B",
                  "in the cafeteria", "on the third floor"],
    },
    "account": {
        "subject": ["my password", "my student account", "my email login",
                    "my university account", "two-factor authentication"],
        "problem": ["has expired and I cannot reset it", "is locked after too many attempts",
                    "is not accepted anymore", "keeps asking for a verification code",
                    "needs to be reset"],
        "place": ["on the student portal", "on webmail", "on my phone",
                  "on the learning platform", ""],
    },
    "printing": {
        "subject": ["the printer", "the library printer", "my print job",
                    "the lab printer", "the copier"],
        "problem": ["is jammed", "prints blank pages", "is out of toner",
                    "shows offline", "is stuck in the queue"],
        "place": ["on floor 2", "in room 204", "near the entrance",
                  "in the computer lab", ""],
    },
    "software": {
        "subject": ["MATLAB", "the office suite", "the VPN client", "Zoom",
                    "the statistics software"],
        "problem": ["will not install", "crashes on startup", "says my license expired",
                    "needs an update but fails", "shows an error on launch"],
        "place": ["on my laptop", "on Windows 11", "on my Mac",
                  "on the lab computer", ""],
    },
    "hardware": {
        "subject": ["my laptop", "the monitor", "my keyboard", "the desktop computer",
                    "my laptop charger"],
        "problem": ["will not turn on", "has a cracked screen", "is overheating",
                    "has keys that stopped working", "makes a loud noise"],
        "place": ["in my office", "in the lab", "at home", "since I dropped it", ""],
    },
}

# Cross-category distractor phrases make the data less trivially separable.
DISTRACTORS = [
    "Also the wifi was slow yesterday.",
    "I tried restarting my laptop.",
    "I think my password is fine.",
    "I printed the error message.",
    "I installed an update last week.",
]


def _make_ticket(rng, label):
    t = TEMPLATES[label]
    body = f"{rng.choice(t['subject'])} {rng.choice(t['problem'])} {rng.choice(t['place'])}".strip()
    body = body[0].upper() + body[1:] + "."
    parts = [rng.choice(OPENERS), body, rng.choice(FILLER)]
    if rng.random() < 0.25:
        parts.append(rng.choice(DISTRACTORS))
    return " ".join(p for p in parts if p).strip()


def build_train(n_per_class=40, seed=42):
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        for _ in range(n_per_class):
            rows.append((_make_ticket(rng, label), label))
    rng.shuffle(rows)
    return rows


# 20 hand-written tickets, never used for training.
CHALLENGE = [
    ("I can't log in to Canvas, it says my credentials are invalid", "account"),
    ("Forgot my password again, how do I get back in?", "account"),
    ("My account got suspended after I changed my phone number", "account"),
    ("Laptop battery drains in 30 minutes and the case gets very hot", "hardware"),
    ("Screen flickers and then goes black when I open the lid", "hardware"),
    ("The mouse pointer freezes, I think the trackpad is dead", "hardware"),
    ("Nothing comes out when I hit print, the job just sits there", "printing"),
    ("Printer on floor 3 keeps eating paper", "printing"),
    ("Can't print my thesis, printer says offline even though wifi is fine", "printing"),
    ("Need Adobe Acrobat installed on my work PC", "software"),
    ("Python keeps throwing an error when I launch the IDE", "software"),
    ("The VPN app closes immediately after I open it", "software"),
    ("No signal at all in the east wing, my phone and laptop both fail", "wifi"),
    ("Connection is super laggy during online exams", "wifi"),
    ("Can't get on the guest network with the code you gave me", "wifi"),
    ("Zoom is slow and the video freezes, internet seems bad", "wifi"),
    ("Windows will not boot, just a black screen with a blinking cursor", "hardware"),
    ("Two-factor codes never arrive on my phone", "account"),
    ("Toner light is red and the pages come out streaky", "printing"),
    ("Installer for the stats package stops at 60 percent", "software"),
]
