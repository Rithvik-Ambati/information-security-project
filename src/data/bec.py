"""Combinatorial BEC (business-email-compromise) generator: opener + ask + justification + closer, in one of
three tones. Written by hand (not produced by an LLM API) to give thousands of distinct messages instead of
a handful of templates.

Leakage control: the *asks* (the part that carries the fraud intent) are disjoint across the train / val / test
eras; openers, justifications and closers are shared, as they would be in real attacker emails."""

OPENERS = ["Hi,", "Hello,", "Hey,", "Good morning,", "Quick one,", "Hope you're well.", "Are you at your desk?",
           "Do you have a minute?", "Sorry to bother you.", "Checking in,", "Are you free right now?",
           "Got a second?", "Apologies for the short notice.", "I need your help with something.",
           "Following up on our earlier chat.", "Just landed and short on time.", "Thanks for getting back to me.",
           "Hope the week is treating you well.", "Hi there,", "Morning,"]

ASKS = {
    "train": [
        "Please wire ${amt} to the new vendor account today and confirm once it is sent.",
        "Can you buy {n} gift cards, ${amt} each, and send me the codes? I will reimburse you.",
        "I changed banks. Please update my direct deposit before {day} payroll.",
        "I am locked out of the {thing}. Can you send me the login for it?",
        "Please process the attached invoice for ${amt} using the updated remittance details.",
        "I need the {thing} credentials and a transfer of ${amt} handled today.",
        "Could you pay ${amt} to a contractor on my behalf? Details to follow once you reply.",
        "Send me the latest payroll file with employee bank details, please.",
        "Please release the pending payment of ${amt} to the account I will send you.",
        "Can you reset the password on the {thing} and send me the temporary one?",
    ],
    "val": [
        "Our supplier changed their remittance details, so could you process ${amt} to the new account by {day}?",
        "I cannot log in to the {thing} from here. Can you pull up the access details for me?",
        "Please arrange {n} prepaid cards worth ${amt} in total for client gifts and message me the numbers.",
        "The CEO asked me to get the {thing} password reset right away. Can you send the temporary credentials?",
        "Please settle the vendor balance of ${amt} today, paperwork will follow.",
        "Do not mention this to anyone yet, but I need a confidential transfer of ${amt}.",
        "Can you forward the staff list with direct deposit information for the {thing}?",
    ],
    "test": [
        "Finance needs the {thing} export with bank details before {day}. Could you send it over?",
        "We are late on a settlement. Release ${amt} now and I will sort out the paperwork afterwards.",
        "I'm travelling and my laptop is misbehaving. Could you share the {thing} access details?",
        "Please pick up {n} iTunes or Amazon cards (${amt} each) as a surprise for the team and send me photos of the codes.",
        "Our accountant sent new account numbers. Please update the vendor record and pay ${amt} there.",
        "Please keep this between us: I need ${amt} moved to a holding account today.",
        "Could you re-send the {thing} login? I will explain later, it is time sensitive.",
    ],
}

JUSTIFY = ["I am in a meeting and cannot talk.", "I will explain later.", "It is time sensitive.",
           "This is for a client, so it is urgent.", "I am travelling and have limited access.",
           "Finance is waiting on this.", "It needs to go out before end of day.", "", "", "",
           "I will make sure you are covered.", "Do not call, email is best right now."]

CLOSERS = {"formal": ["Regards,", "Kind regards,", "Best regards,", "Sincerely,"],
           "casual": ["Thanks!", "Cheers,", "Thx,", "Appreciate it,"],
           "terse": ["", "Thanks", "-", ""]}


def bec_email(rng, era):
    tone = rng.choice(["formal", "casual", "terse"])
    ask = rng.choice(ASKS[era]).format(
        amt=rng.randint(4, 120) * 100, n=rng.randint(3, 9),
        day=rng.choice(["Friday", "Monday", "Thursday", "end of day", "tomorrow morning"]),
        thing=rng.choice(["vendor portal", "payroll system", "bank portal", "finance drive", "HR system",
                          "expense tool", "treasury dashboard"]))
    parts = [rng.choice(OPENERS), ask, rng.choice(JUSTIFY), rng.choice(CLOSERS[tone])]
    text = " ".join(p for p in parts if p)
    if tone == "terse":
        text = text.lower().replace("  ", " ")
    return text
