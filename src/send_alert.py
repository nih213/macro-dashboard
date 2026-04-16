"""
Sends a recession probability alert email when the probability crosses a threshold.
Called by GitHub Actions after build_cache.py completes.

Required GitHub Actions secrets:
  SMTP_USER         Gmail address to send from  (e.g. yourname@gmail.com)
  SMTP_PASSWORD     Gmail App Password (Settings > Security > App passwords)
  SUBSCRIBER_REPO   Private GitHub repo storing subscribers.txt  (e.g. you/macro-dashboard-data)
  PAT_TOKEN         Personal access token with repo scope (to read the private subscriber repo)

Optional:
  ALERT_EMAILS      Comma-separated fallback recipients if SUBSCRIBER_REPO is not configured
  ALERT_THRESHOLD   Probability % that triggers an alert (default: 25)
"""
import os
import base64
import pickle
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_PATH   = os.path.join(PROJECT_ROOT, "data", "cache.pkl")


def fetch_subscribers(repo: str, token: str) -> list[str]:
    """Fetch subscriber emails from data/subscribers.txt in the private GitHub repo."""
    url     = f"https://api.github.com/repos/{repo}/contents/data/subscribers.txt"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        content = base64.b64decode(r.json()["content"]).decode()
        return [e.strip() for e in content.splitlines() if e.strip()]
    print(f"Warning: could not fetch subscribers ({r.status_code}). Continuing without file list.")
    return []


def send():
    smtp_user  = os.environ.get("SMTP_USER", "")
    smtp_pass  = os.environ.get("SMTP_PASSWORD", "")
    threshold  = float(os.environ.get("ALERT_THRESHOLD") or "25")

    # Collect recipients: private repo subscribers + optional ALERT_EMAILS fallback
    sub_repo  = os.environ.get("SUBSCRIBER_REPO", "")
    pat       = os.environ.get("PAT_TOKEN", "")
    file_emails = fetch_subscribers(sub_repo, pat) if sub_repo and pat else []
    env_emails  = [e.strip() for e in os.environ.get("ALERT_EMAILS", "").split(",") if e.strip()]
    recipients  = list({*file_emails, *env_emails})   # deduplicate

    if not smtp_user or not smtp_pass or not recipients:
        print("Email alert skipped: SMTP credentials not set, or no subscribers.")
        return

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    prob      = cache["prob_series"].iloc[-1]
    prev_prob = cache.get("prev_prob")
    date      = cache["prob_series"].index[-1].strftime("%B %Y")

    # Only send on threshold crossings, not every run
    if prev_prob is not None:
        if (prev_prob >= threshold) == (prob >= threshold):
            print(f"No crossing (prev={prev_prob:.1f}%, now={prob:.1f}%, threshold={threshold:.0f}%). No alert sent.")
            return

    direction = "crossed above" if prob >= threshold else "dropped below"
    change    = prob - prev_prob if prev_prob is not None else 0
    forecast  = cache["prob_series"].index[-1]
    import pandas as pd
    forecast_month = (forecast + pd.DateOffset(months=3)).strftime("%B %Y")

    subject = f"Recession Alert: probability {direction} {threshold:.0f}% — now {prob:.1f}%"

    body = f"""
<html><body style="font-family:sans-serif; color:#1e293b; max-width:600px; margin:auto; padding:24px">
<h2 style="color:#1e293b">U.S. Recession Probability Alert</h2>

<div style="padding:20px; border-radius:8px; background:#f8fafc; border:1px solid #e2e8f0; text-align:center; margin:20px 0">
  <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em">Recession Probability by {forecast_month}</div>
  <div style="font-size:48px; font-weight:800; color:{'#ef4444' if prob >= 50 else '#eab308' if prob >= 20 else '#22c55e'}">{prob:.1f}%</div>
  <div style="font-size:13px; color:#64748b">Based on data as of {date}</div>
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
You subscribed to U.S. Recession Probability alerts.<br>
To unsubscribe, reply to this email.
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
