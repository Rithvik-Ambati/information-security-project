"""Plain-English explanations and a mock mail-client warning banner (HTML) from trust reasons + SHAP drivers."""
import html

FEATURE_TEXT = {
    "text_score": "The wording closely resembles known phishing / business-email-compromise messages",
    "anomaly": "This message is out of character compared with the sender's normal behavior",
    "urgent": "Uses urgent or threatening language",
    "payment": "Asks for money, gift cards or credentials",
    "payment_novel": "Requests payment/credentials, which this sender has never done before",
    "url_novel": "Contains a link, which this sender rarely sends",
    "url_odd": "Contains a suspicious-looking link",
    "n_urls": "Contains links",
    "odd_hour": "Sent at an unusual hour (night-time)",
    "hour_typical": "Sent at an hour this sender does not normally write",
    "new_recipient": "First time this sender writes to this recipient",
    "rcpt_domain_new": "Addressed to a domain this sender has never written to",
    "rcpt_prior_count": "Little prior correspondence with this recipient",
    "style_sim": "Does not sound like this sender (writing style differs from their history)",
    "wordlen_dev": "Word choice differs from the sender's usual style",
    "punct_dev": "Punctuation habits differ from the sender's usual style",
    "greet_dev": "Greeting habits differ from the sender's usual style",
    "len_ratio": "Length is unusual for this sender",
    "caps_dev": "Capitalisation differs from the sender's usual style",
    "lookalike_domain": "Sender domain imitates a domain you already trust",
    "rp_mismatch": "The 'From' and 'Return-Path' domains do not match",
    "unsigned": "The message is not cryptographically signed",
    "sender_known": "First contact from this sender",
    "domain_known": "Sender domain has not been seen before",
    "prior_urgent_rate": "Urgent tone is unusual for this sender",
    "n_received": "Unusual mail routing",
}


def drivers_to_text(why: dict, k=4):
    return [FEATURE_TEXT.get(f, f.replace("_", " ")) for f in list(why)[:k]]


def banner_html(label, sender, subject, body, trust_reasons, why, trust, risk, action):
    """Gmail-style message view with a coloured warning banner on top."""
    color = {"Malicious": ("#fce8e6", "#c5221f", "&#9888; This message looks dangerous"),
             "Suspicious": ("#fef7e0", "#b06000", "&#9888; Be careful with this message"),
             "Trusted": ("#e6f4ea", "#137333", "&#10003; No threats detected")}[label]
    reasons = list(trust_reasons) + drivers_to_text(why)
    items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons) or "<li>No risk indicators found</li>"
    return f"""
<div style="font-family:Arial,sans-serif;border:1px solid #dadce0;border-radius:8px;max-width:760px;overflow:hidden">
  <div style="background:{color[0]};color:{color[1]};padding:12px 16px;border-bottom:1px solid #dadce0">
    <b style="font-size:15px">{color[2]}</b>
    <div style="font-size:13px;margin-top:6px">Action taken: <b>{html.escape(action)}</b> &nbsp;|&nbsp;
    Trust {trust:.2f} &nbsp;|&nbsp; Risk {risk:.2f}</div>
    <div style="font-size:13px;margin-top:6px">Why we flagged it:</div>
    <ul style="margin:4px 0 0 18px;font-size:13px">{items}</ul>
  </div>
  <div style="padding:14px 16px;color:#202124">
    <div style="font-size:18px;margin-bottom:6px">{html.escape(subject or '(no subject)')}</div>
    <div style="font-size:13px;color:#5f6368;margin-bottom:10px">From: {html.escape(sender)}</div>
    <div style="font-size:14px;line-height:1.5">{html.escape(body[:500])}</div>
  </div>
</div>"""
