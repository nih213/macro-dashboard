"""
Sends a recession probability alert email when the probability crosses a threshold.
Called by GitHub Actions after build_cache.py completes.

Required GitHub Actions secrets:
  SMTP_USER       Gmail address to send from  (e.g. yourname@gmail.com)
  SMTP_PASSWORD   Gmail App Password (Settings > Security > App passwords)
  ALERT_EMAILS    Comma-separated recipient list (e.g. friend1@gmail.com,friend2@gmail.com)

Optional:
  ALERT_THRESHOLD  Probability % that triggers an alert (default: 25)
"""
import os
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_PATH   = os.path.join(PROJECT_ROOT, "data", "cache.pkl")


def send():
    smtp_user  = os.environ.get("SMTP_USER", "")
    smtp_pass  = os.environ.get("SMTP_PASSWORD", "")
    recipients = [e.strip() for e in os.environ.get("ALERT_EMAILS", "").split(",") if e.strip()]
    threshold  = float(os.environ.get("ALERT_THRESHOLD", "25"))

    # Merge subscribers file + env var override
    subs_path = os.path.join(PROJECT_ROOT, "data", "subscribers.txt")
    if os.path.exists(subs_path):
        with open(subs_path) as f:
            file_emails = [e.strip() for e in f if e.strip()]
    else:
        file_emails = []
    recipients = list({*file_emails, *recipients})   # deduplicate

    if not smtp_user or not smtp_pass or not recipients:
        print("Email alert skipped: SMTP_USER, SMTP_PASSWORD not set, or no subscribers.")
        return

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    prob      = cache["prob_series"].iloc[-1]
    prev_prob = cache.get("prev_prob")
    date      = cache["prob_series"].index[-1].strftime("%B %Y")

    # Only send on threshold crossings, not every day
    if prev_prob is not None:
        was_above = prev_prob >= threshold
        now_above = prob >= threshold
        if was_above == now_above:
            print(f"No crossing (prev={prev_prob:.1f}%, now={prob:.1f}%, threshold={threshold:.0f}%). No alert sent.")
            return

    direction = "crossed above" if prob >= threshold else "dropped below"
    change    = prob - prev_prob if prev_prob is not None else 0
    subject   = f"Recession Alert: probability {direction} {threshold:.0f}% — now {prob:.1f}%"

    body = f"""
<html><body style="font-family:sans-serif; color:#1e293b; max-width:600px; margin:auto; padding:24px">
<h2 style="color:#1e293b">U.S. Recession Probability Alert</h2>

<div style="padding:20px; border-radius:8px; background:#f8fafc; border:1px solid #e2e8f0; text-align:center; margin:20px 0">
  <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em">3-Month Recession Probability</div>
  <div style="font-size:48px; font-weight:800; color:{'#ef4444' if prob >= 50 else '#eab308' if prob >= 20 else '#22c55e'}">{prob:.1f}%</div>
  <div style="font-size:13px; color:#64748b">As of {date}</div>
</div>

<p>The recession probability has <strong>{direction} the {threshold:.0f}% alert threshold</strong>.</p>
{"<p>Previous reading: " + f"{prev_prob:.1f}% &nbsp;→&nbsp; {prob:.1f}% &nbsp;({change:+.1f} pp)" + "</p>" if prev_prob is not None else ""}

<p style="color:#475569; font-size:14px">
The model is a logistic regression trained on 14 leading macro indicators
(yield curve, credit spreads, industrial production, payrolls, monetary policy, and more),
forecasting the probability of a U.S. recession within the next three months.
</p>

<hr style="border:none; border-top:1px solid #e2e8f0; margin:20px 0">
<p style="color:#94a3b8; font-size:12px">
You are receiving this alert because your email is on the ALERT_EMAILS list.<br>
To unsubscribe, remove your address from the ALERT_EMAILS secret in the GitHub repository.
</p>
</body></html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipients, msg.as_string())

    print(f"Alert sent to {len(recipients)} recipient(s): {subject}")


if __name__ == "__main__":
    send()
