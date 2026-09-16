import html
import os

import google.auth
import google.auth.transport.requests
import requests
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]
ENDPOINT_NAME = os.environ["ENDPOINT_DISPLAY_NAME"]

PREDICT_URL = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/{REGION}/endpoints/{ENDPOINT_NAME}:predict"
)

app = FastAPI()


def classify(headline: str) -> dict:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())

    response = requests.post(
        PREDICT_URL,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json={"instances": [{"text": headline}]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["predictions"][0]


def render_page(headline: str = "", result_html: str = "") -> str:
    safe_headline = html.escape(headline, quote=True)
    return f"""<!doctype html>
<html>
<head>
<title>Headline Classifier</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 3rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  form {{ display: flex; gap: 0.5rem; margin: 1.5rem 0; }}
  input[type=text] {{ flex: 1; padding: 0.6rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }}
  button {{ padding: 0.6rem 1.2rem; font-size: 1rem; border: 0; border-radius: 6px; background: #1a73e8; color: white; cursor: pointer; }}
  button:hover {{ background: #1558b3; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ padding: 0.3rem 0.5rem; border-bottom: 1px solid #eee; }}
  td.prob {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .predicted {{ font-size: 1.1rem; margin-bottom: 0.5rem; }}
  .error {{ color: #b00020; }}
  .top td {{ font-weight: 600; }}
</style>
</head>
<body>
  <h1>News Headline Classifier</h1>
  <p>Trained on AG News - predicts World / Sports / Business / Sci-Tech.</p>
  <form method="post">
    <input type="text" name="headline" placeholder="Enter a news headline..." value="{safe_headline}" autofocus>
    <button type="submit">Classify</button>
  </form>
  {result_html}
</body>
</html>"""


def render_result(prediction: dict) -> str:
    probabilities = sorted(
        prediction["probabilities"].items(), key=lambda kv: -kv[1]
    )
    rows = "\n".join(
        f'    <tr class="{"top" if cls == prediction["predicted_class"] else ""}">'
        f"<td>{html.escape(cls)}</td><td class=\"prob\">{prob * 100:.1f}%</td></tr>"
        for cls, prob in probabilities
    )
    return f"""
  <div class="result">
    <p class="predicted">Predicted: <strong>{html.escape(prediction['predicted_class'])}</strong></p>
    <table>
{rows}
    </table>
  </div>"""


def render_error(message: str) -> str:
    return f'<p class="error">Error: {html.escape(message)}</p>'


@app.get("/", response_class=HTMLResponse)
def index():
    return render_page()


@app.post("/", response_class=HTMLResponse)
def predict(headline: str = Form(...)):
    headline = headline.strip()
    if not headline:
        return render_page(result_html=render_error("Please enter a headline."))
    try:
        prediction = classify(headline)
    except Exception as e:
        return render_page(headline=headline, result_html=render_error(str(e)))
    return render_page(headline=headline, result_html=render_result(prediction))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
