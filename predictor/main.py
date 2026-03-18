"""
K8s Autopilot - Prediction Service API
"""
import os
import json
import time
import logging
import threading
from contextlib import asynccontextmanager
from typing import List, Optional
from dataclasses import asdict

import redis
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from prometheus_client import make_asgi_app, Gauge, Counter

from predictor import AutopilotPredictor, PodPrediction
from fetcher import fetch_pod_windows

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [PREDICTOR] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

REDIS_URL           = os.getenv("REDIS_URL", "redis://redis:6379")
PREDICTION_INTERVAL = int(os.getenv("PREDICTION_INTERVAL", "30"))
API_KEY             = os.getenv("API_KEY", "")
PREDICTION_KEY      = "autopilot:predictions"
HISTORY_KEY         = "autopilot:history"
METRICS_KEY         = "autopilot:model_metrics"
STATUS_KEY          = "autopilot:status"

prom_predictions_total = Counter("autopilot_predictions_total", "Total predictions", ["risk_level"])
prom_high_risk_pods    = Gauge("autopilot_high_risk_pods", "High risk pod count")

predictor            = AutopilotPredictor()
_r_pool: Optional[redis.ConnectionPool] = None
_r_client: Optional[redis.Redis]        = None
current_predictions: List[PodPrediction] = []
prediction_lock      = threading.Lock()
_last_retrain_time   = 0.0
RETRAIN_COOLDOWN     = 120


# ── AUTH ──────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        return True
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True


# ── REDIS — robust reconnection ───────────────────────────────────
def get_redis() -> Optional[redis.Redis]:
    """Return a Redis client, reconnecting if the connection has dropped."""
    global _r_pool, _r_client
    if _r_client is not None:
        try:
            _r_client.ping()
            return _r_client
        except Exception:
            _r_pool   = None
            _r_client = None

    try:
        _r_pool   = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
        _r_client = redis.Redis(connection_pool=_r_pool, socket_timeout=3)
        _r_client.ping()
        return _r_client
    except Exception as e:
        log.warning(f"Redis unavailable: {e}")
        _r_client = None
        return None


def _set_status(status: str, rc: Optional[redis.Redis]):
    if rc:
        try:
            rc.set(STATUS_KEY, status, ex=600)
        except redis.RedisError:
            pass


def store_predictions(predictions):
    data = [asdict(p) for p in predictions]
    rc = get_redis()
    if rc:
        try:
            rc.set(PREDICTION_KEY, json.dumps(data), ex=300)
            for p in predictions:
                if p.risk_level in ("high", "critical"):
                    rc.lpush(HISTORY_KEY, json.dumps(asdict(p)))
            rc.ltrim(HISTORY_KEY, 0, 999)
        except redis.RedisError as e:
            log.error(f"Redis store failed: {e}")
    return data


def load_predictions():
    rc = get_redis()
    if rc:
        try:
            raw = rc.get(PREDICTION_KEY)
            if raw:
                return json.loads(raw)
        except redis.RedisError:
            pass
    with prediction_lock:
        return [asdict(p) for p in current_predictions]


# ── PREDICTION LOOP ───────────────────────────────────────────────
def prediction_loop():
    global current_predictions

    rc = get_redis()
    _set_status("training", rc)
    log.info("Training model with synthetic data...")
    training_data = predictor.generate_synthetic_training_data(n_normal=600, n_failure=150)
    metrics = predictor.train(training_data)
    log.info(f"Training complete: {metrics}")

    rc = get_redis()
    if rc:
        try:
            rc.set(METRICS_KEY, json.dumps({**metrics, "status": "trained"}), ex=86400)
        except redis.RedisError:
            pass
    _set_status("running", rc)

    while True:
        try:
            windows = fetch_pod_windows()

            # FIX: surface warm-up state when Prometheus hasn't accumulated data yet
            if not windows:
                log.info("No pod windows yet — Prometheus may still be warming up")
                _set_status("warming_up", get_redis())
                time.sleep(PREDICTION_INTERVAL)
                continue

            _set_status("running", get_redis())
            predictions = []

            for window in windows:
                pred = predictor.predict(window)
                if pred:
                    predictions.append(pred)
                    prom_predictions_total.labels(risk_level=pred.risk_level).inc()

            order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            predictions.sort(key=lambda p: (order.get(p.risk_level, 4), -p.risk_score))

            with prediction_lock:
                current_predictions = predictions

            store_predictions(predictions)
            high_risk = sum(1 for p in predictions if p.risk_level in ("high", "critical"))
            prom_high_risk_pods.set(high_risk)
            log.info(f"Predictions done: {len(predictions)} pods, {high_risk} high-risk")

        except Exception as e:
            log.error(f"Prediction loop error: {e}", exc_info=True)

        time.sleep(PREDICTION_INTERVAL)


# ── FASTAPI APP ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    threading.Thread(target=prediction_loop, daemon=True).start()
    log.info("Predictor API ready")
    yield
    log.info("Predictor shutting down")


app = FastAPI(title="K8s Autopilot Predictor", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    rc = get_redis()
    status = "unknown"
    if rc:
        try:
            status = rc.get(STATUS_KEY) or "running"
        except redis.RedisError:
            pass
    return {
        "status": "ok",
        "predictor_status": status,
        "trained": predictor.is_trained,
        "metrics": predictor.metrics,
    }


@app.get("/predictions")
def get_predictions(risk_level: Optional[str] = None, namespace: Optional[str] = None):
    data = load_predictions()
    if risk_level:
        data = [p for p in data if p.get("risk_level") == risk_level]
    if namespace:
        data = [p for p in data if p.get("namespace") == namespace]
    return {"predictions": data, "count": len(data), "timestamp": time.time()}


@app.get("/predictions/summary")
def get_summary():
    data = load_predictions()
    levels = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    causes = {"oom": 0, "cpu": 0, "network": 0, "unknown": 0}
    for p in data:
        rl = p.get("risk_level", "low")
        pc = p.get("predicted_cause", "unknown")
        levels[rl] = levels.get(rl, 0) + 1
        causes[pc] = causes.get(pc, 0) + 1
    return {
        "total_pods": len(data),
        "by_risk_level": levels,
        "by_cause": causes,
        "model_metrics": predictor.metrics,
    }


@app.get("/history")
def get_history(limit: int = 50):
    rc = get_redis()
    if not rc:
        return {"events": [], "count": 0}
    try:
        raw    = rc.lrange(HISTORY_KEY, 0, limit - 1)
        events = [json.loads(e) for e in raw]
        return {"events": events, "count": len(events)}
    except redis.RedisError:
        return {"events": [], "count": 0}


@app.post("/retrain", dependencies=[Depends(verify_api_key)])
def retrain():
    global _last_retrain_time
    now = time.time()
    if now - _last_retrain_time < RETRAIN_COOLDOWN:
        remaining = int(RETRAIN_COOLDOWN - (now - _last_retrain_time))
        raise HTTPException(
            status_code=429,
            detail=f"Retrain on cooldown. Try again in {remaining}s",
        )
    _last_retrain_time = now
    training_data = predictor.generate_synthetic_training_data(600, 150)
    metrics       = predictor.train(training_data)
    rc = get_redis()
    if rc:
        try:
            rc.set(METRICS_KEY, json.dumps({**metrics, "status": "trained"}), ex=86400)
        except redis.RedisError:
            pass
    return {"status": "retrained", "metrics": metrics}


@app.get("/model/metrics")
def model_metrics():
    rc = get_redis()
    if rc:
        try:
            raw = rc.get(METRICS_KEY)
            if raw:
                return json.loads(raw)
        except redis.RedisError:
            pass
    return predictor.metrics


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")