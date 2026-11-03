"""Run from backend/:  python check_keys.py
Sends one real request to Jev and one to Gemini and prints the raw responses,
then checks that SignalOS parses the Jev response. Paste the output back."""
import json

import httpx

from app import config
from app.services import jev_service

TEXT = "The app crashes every time I try to pay. I've tried three times. Uninstalling."

print("keys loaded -> jev:", config.jev_live(), "| gemini:", config.gemini_live())

print("\n=== JEV raw response ===")
try:
    r = httpx.post(config.JEV_URL, headers={"Authorization": f"Bearer {config.TYPESAFE_API_KEY}"},
                   json=jev_service.build_request(TEXT), timeout=30)
    print("HTTP", r.status_code)
    print(r.text[:3000])
    if r.status_code == 200:
        print("\n=== SignalOS parsed it as ===")
        print(json.dumps(jev_service.parse_response(r.json()), indent=2))
except Exception as e:
    print("ERROR", type(e).__name__, e)

print("\n=== GEMINI ===")
try:
    r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
                   headers={"x-goog-api-key": config.GEMINI_API_KEY},
                   json={"contents": [{"parts": [{"text": "Reply with exactly: ok"}]}]}, timeout=30)
    print("HTTP", r.status_code)
    print(r.text[:800])
except Exception as e:
    print("ERROR", type(e).__name__, e)
