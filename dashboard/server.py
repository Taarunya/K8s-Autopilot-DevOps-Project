"""
K8s Autopilot - Dashboard Server
Serves the frontend and proxies API calls to backend services.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [DASHBOARD] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

PREDICTOR_URL = os.getenv("PREDICTOR_URL", "http://predictor:8001")
OPERATOR_URL  = os.getenv("OPERATOR_URL",  "http://operator:8002")
API_KEY       = os.getenv("API_KEY", "")

_html_cache: Optional[str] = None


def _load_html() -> str:
    paths = [
        "/app/index.html",
        "index.html",
        os.path.join(os.path.dirname(__file__), "index.html"),
    ]
    for path in paths:
        if os.path.exists(path):
            log.info(f"Loaded index.html from {path}")
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    log.error("index.html NOT FOUND")
    return "<html><body><h1>Dashboard HTML not found</h1></body></html>"


def _auth_headers() -> dict:
    """Return API key headers if a key is configured."""
    return {"X-API-Key": API_KEY} if API_KEY else {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _html_cache
    _html_cache = _load_html()
    log.info("Dashboard server ready")
    yield
    log.info("Dashboard shutting down")


app = FastAPI(title="K8s Autopilot Dashboard", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def proxy_get(url: str, timeout: int = 5) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError as e:
        log.warning(f"Connection failed {url}: {e}")
        return {"error": "service_unavailable"}
    except requests.Timeout:
        log.warning(f"Timeout {url}")
        return {"error": "timeout"}
    except requests.RequestException as e:
        log.warning(f"Request failed {url}: {e}")
        return {"error": str(e)}


def proxy_post(url: str, timeout: int = 60) -> dict:
    try:
        r = requests.post(url, headers=_auth_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError as e:
        log.warning(f"Connection failed {url}: {e}")
        return {"error": "service_unavailable"}
    except requests.Timeout:
        log.warning(f"Timeout {url}")
        return {"error": "timeout"}
    except requests.RequestException as e:
        log.warning(f"Request failed {url}: {e}")
        return {"error": str(e)}


def proxy_delete(url: str, timeout: int = 10) -> dict:
    """FIX: centralised DELETE helper — uses shared _auth_headers() so it can't drift."""
    try:
        r = requests.delete(url, headers=_auth_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError as e:
        log.warning(f"Connection failed {url}: {e}")
        return {"error": "service_unavailable"}
    except requests.Timeout:
        log.warning(f"Timeout {url}")
        return {"error": "timeout"}
    except requests.RequestException as e:
        log.warning(f"Request failed {url}: {e}")
        return {"error": str(e)}


# ── API ROUTES ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=_html_cache or _load_html())


@app.get("/api/predictions")
def predictions(risk_level: str = None, namespace: str = None):
    url    = f"{PREDICTOR_URL}/predictions"
    params = []
    if risk_level:
        params.append(f"risk_level={risk_level}")
    if namespace:
        params.append(f"namespace={namespace}")
    if params:
        url += "?" + "&".join(params)
    return proxy_get(url)


@app.get("/api/summary")
def summary():
    return {
        "predictions": proxy_get(f"{PREDICTOR_URL}/predictions/summary"),
        "actions":     proxy_get(f"{OPERATOR_URL}/actions/summary"),
        "health":      proxy_get(f"{PREDICTOR_URL}/health"),
    }


@app.get("/api/actions")
def actions(limit: int = 50):
    return proxy_get(f"{OPERATOR_URL}/actions?limit={limit}")


@app.get("/api/history")
def history(limit: int = 30):
    return proxy_get(f"{PREDICTOR_URL}/history?limit={limit}")


@app.get("/api/model")
def model():
    return proxy_get(f"{PREDICTOR_URL}/model/metrics")


@app.post("/api/retrain")
def retrain():
    return proxy_post(f"{PREDICTOR_URL}/retrain")


@app.delete("/api/clear-actions")
def clear_actions():
    # FIX: use shared proxy_delete helper — consistent auth + error handling
    return proxy_delete(f"{OPERATOR_URL}/actions/clear")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")