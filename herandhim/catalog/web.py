"""Serve only public catalogue assets. Run separately from the private dashboard.

    uvicorn herandhim.catalog.web:app --host 127.0.0.1 --port 7789
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .characters import CHARACTERS

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def public_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' https://telegram.org; "
        "style-src 'self'; img-src 'self'; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; "
        "frame-ancestors https://web.telegram.org https://*.telegram.org"
    )
    return response


@app.get("/catalog.json")
def catalogue():
    return {"version": 1, "title": "AIGirla", "bot_url": "https://t.me/sara_herandhim_bot",
            "characters": [item.public() for item in CHARACTERS]}


@app.get("/healthz")
def health():
    return {"status": "ok", "surface": "public_catalog"}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "web/static/catalog", html=True), name="catalog")
