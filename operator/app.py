"""
K8s Autopilot - Production Operator
Real Kubernetes API + Slack alerts + proper error handling
"""
import os
import json
import time
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

import redis
import requests
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from dataclasses import dataclass, asdict
from prometheus_client import make_asgi_app, Counter, Gauge, Histogram

try:
    from kubernetes import client as k8s_client, config as k8s_config
    K8S_LIB_AVAILABLE = True
except ImportError:
    K8S_LIB_AVAILABLE = False

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [OPERATOR] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────
PREDICTOR_URL  = os.getenv("PREDICTOR_URL",  "http://predictor:8001")
REDIS_URL      = os.getenv("REDIS_URL",      "redis://redis:6379")
MOCK_K8S       = os.getenv("MOCK_K8S",       "true").lower() == "true"
AUTO_REMEDIATE = os.getenv("AUTO_REMEDIATE", "true").lower() == "true"
DRY_RUN        = os.getenv("DRY_RUN",        "false").lower() == "true"
SLACK_WEBHOOK  = os.getenv("SLACK_WEBHOOK_URL", "")
LOOP_INTERVAL  = int(os.getenv("LOOP_INTERVAL", "30"))
COOLDOWN_SECS  = int(os.getenv("COOLDOWN_SECS", "300"))
API_KEY        = os.getenv("API_KEY", "")

ACTIONS_KEY  = "autopilot:actions"
COOLDOWN_KEY = "autopilot:cooldown"
STATS_KEY    = "autopilot:stats"

# ── PROMETHEUS METRICS ────────────────────────────────────────────
prom_actions_total  = Counter("autopilot_operator_actions_total", "Total remediation actions", ["action_type", "result"])
prom_cycle_duration = Histogram("autopilot_operator_cycle_seconds", "Duration of each operator cycle")
prom_active_pods    = Gauge("autopilot_operator_monitored_pods", "Number of pods being monitored")
prom_errors_total   = Counter("autopilot_operator_errors_total", "Total operator errors", ["component"])


@dataclass
class RemediationAction:
    pod: str
    namespace: str
    deployment: str
    node: str
    action_type: str
    risk_level: str
    risk_score: float
    predicted_cause: str
    eta_minutes: Optional[int]
    dry_run: bool
    result: str
    timestamp: float
    detail: str


# ── REDIS (connection-pooled, auto-reconnect) ─────────────────────
_redis_pool: Optional[redis.ConnectionPool] = None
_r_client:   Optional[redis.Redis]          = None


def get_redis() -> Optional[redis.Redis]:
    global _redis_pool, _r_client
    if _r_client is not None:
        try:
            _r_client.ping()
            return _r_client
        except Exception:
            _redis_pool = None
            _r_client   = None

    try:
        _redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
        _r_client   = redis.Redis(connection_pool=_redis_pool, socket_timeout=3)
        _r_client.ping()
        return _r_client
    except Exception as e:
        log.warning(f"Redis unavailable: {e}")
        _r_client = None
        return None


def is_cooling_down(pod: str, rc: Optional[redis.Redis]) -> bool:
    if not rc:
        return False
    try:
        return bool(rc.get(f"{COOLDOWN_KEY}:{pod}"))
    except redis.RedisError as e:
        log.debug(f"Cooldown check failed for {pod}: {e}")
        return False


def set_cooldown(pod: str, rc: Optional[redis.Redis]):
    if rc:
        try:
            rc.set(f"{COOLDOWN_KEY}:{pod}", "1", ex=COOLDOWN_SECS)
        except redis.RedisError as e:
            log.warning(f"Failed to set cooldown for {pod}: {e}")


def store_action(action: RemediationAction, rc: Optional[redis.Redis]):
    if rc:
        try:
            rc.lpush(ACTIONS_KEY, json.dumps(asdict(action)))
            rc.ltrim(ACTIONS_KEY, 0, 999)
            rc.hincrby(STATS_KEY, "total", 1)
            rc.hincrby(STATS_KEY, action.result, 1)
        except redis.RedisError as e:
            log.warning(f"Failed to store action: {e}")


# ── KUBERNETES API ────────────────────────────────────────────────
_k8s_apps_v1: Optional[object] = None
_k8s_core_v1: Optional[object] = None


def init_k8s_client():
    global _k8s_apps_v1, _k8s_core_v1
    if not K8S_LIB_AVAILABLE:
        log.warning("kubernetes library not available, using mock mode")
        return
    try:
        try:
            k8s_config.load_incluster_config()
            log.info("Loaded in-cluster K8s config")
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
            log.info("Loaded kubeconfig from file")
        _k8s_apps_v1 = k8s_client.AppsV1Api()
        _k8s_core_v1 = k8s_client.CoreV1Api()
    except Exception as e:
        log.warning(f"K8s client init failed: {e} — real actions will fail")


def real_scale_deployment(namespace: str, deployment: str, factor: float = 1.5) -> str:
    if not _k8s_apps_v1:
        return "K8s client not initialized"
    try:
        dep         = _k8s_apps_v1.read_namespaced_deployment(deployment, namespace)
        current     = dep.spec.replicas or 1
        new_replicas = max(current + 1, int(current * factor))
        dep.spec.replicas = new_replicas
        _k8s_apps_v1.patch_namespaced_deployment(deployment, namespace, dep)
        return f"Scaled {deployment}: {current} -> {new_replicas} replicas"
    except Exception as e:
        return f"Scale failed: {e}"


def real_rolling_restart(namespace: str, deployment: str) -> str:
    if not _k8s_apps_v1:
        return "K8s client not initialized"
    try:
        patch = {"spec": {"template": {"metadata": {"annotations": {
            "kubectl.kubernetes.io/restartedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }}}}}
        _k8s_apps_v1.patch_namespaced_deployment(deployment, namespace, patch)
        return f"Rolling restart triggered: {deployment}"
    except Exception as e:
        return f"Restart failed: {e}"


def real_cordon_node(node: str) -> str:
    if not _k8s_core_v1:
        return "K8s client not initialized"
    try:
        _k8s_core_v1.patch_node(node, {"spec": {"unschedulable": True}})
        return f"Node {node} cordoned"
    except Exception as e:
        return f"Cordon failed: {e}"


def real_bump_memory(namespace: str, deployment: str, pct: float = 0.25) -> str:
    if not _k8s_apps_v1:
        return "K8s client not initialized"
    try:
        dep        = _k8s_apps_v1.read_namespaced_deployment(deployment, namespace)
        containers = dep.spec.template.spec.containers
        if not containers:
            return "No containers found"
        c      = containers[0]
        limits = c.resources.limits if c.resources and c.resources.limits else {}
        mem    = limits.get("memory", "256Mi")
        if mem.endswith("Mi"):
            new_mem = f"{int(int(mem[:-2]) * (1 + pct))}Mi"
        elif mem.endswith("Gi"):
            new_mem = f"{round(float(mem[:-2]) * (1 + pct), 1)}Gi"
        else:
            return f"Cannot parse memory: {mem}"
        patch = {"spec": {"template": {"spec": {"containers": [
            {"name": c.name, "resources": {"limits": {"memory": new_mem}}}
        ]}}}}
        _k8s_apps_v1.patch_namespaced_deployment(deployment, namespace, patch)
        return f"Memory: {mem} -> {new_mem}"
    except Exception as e:
        return f"Limit bump failed: {e}"


# ── MOCK ACTIONS ──────────────────────────────────────────────────
def mock_scale(ns, dep):
    msg = f"[MOCK] Scale {ns}/{dep} x1.5"; log.info(msg); return msg

def mock_restart(ns, dep):
    msg = f"[MOCK] Restart {ns}/{dep}"; log.info(msg); return msg

def mock_cordon(node):
    msg = f"[MOCK] Cordon {node}"; log.info(msg); return msg

def mock_bump(ns, dep):
    msg = f"[MOCK] Bump limits {ns}/{dep}"; log.info(msg); return msg


# ── SLACK (non-blocking) ──────────────────────────────────────────
def _send_slack_payload(payload: dict):
    """Fire-and-forget Slack POST — runs in its own thread so it never blocks the operator loop."""
    try:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=5).raise_for_status()
        log.info("Slack notification sent")
    except requests.RequestException as e:
        log.warning(f"Slack failed: {e}")
        prom_errors_total.labels(component="slack").inc()


def notify_slack(action: RemediationAction):
    if not SLACK_WEBHOOK:
        return
    emoji  = {"critical": ":rotating_light:", "high": ":warning:", "medium": ":bar_chart:"}.get(action.risk_level, ":info:")
    color  = {"critical": "#ff4757", "high": "#f5a623", "medium": "#0096ff"}.get(action.risk_level, "#6e7681")
    labels = {"scale_up": "Scaled replicas", "restart": "Rolling restart",
               "cordon": "Node cordoned", "limit_bump": "Memory bumped", "alert_only": "Alert only"}
    payload = {
        "attachments": [{
            "color": color,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": f"{emoji} *K8s Autopilot* — `{action.risk_level.upper()}`"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*Pod*\n`{action.pod}`"},
                    {"type": "mrkdwn", "text": f"*Namespace*\n`{action.namespace}`"},
                    {"type": "mrkdwn", "text": f"*Risk*\n{int(action.risk_score * 100)}%"},
                    {"type": "mrkdwn", "text": f"*Cause*\n{action.predicted_cause}"},
                    {"type": "mrkdwn", "text": f"*Action*\n{labels.get(action.action_type, action.action_type)}"},
                    {"type": "mrkdwn", "text": f"*Result*\n{action.result}"},
                ]},
                {"type": "context", "elements": [
                    {"type": "mrkdwn", "text": f"Node: `{action.node}` | {action.detail[:100]}"}
                ]},
            ],
        }]
    }
    # FIX: run in background thread — never block the operator loop on a Slack call
    threading.Thread(target=_send_slack_payload, args=(payload,), daemon=True).start()


# ── DECISION + REMEDIATION ────────────────────────────────────────
def decide_action(pred: dict) -> str:
    risk  = pred.get("risk_level", "low")
    cause = pred.get("predicted_cause", "unknown")
    if risk == "critical":
        if cause == "oom":     return "scale_up"
        if cause == "cpu":     return "cordon"
        return "restart"  # network, unknown
    elif risk == "high":
        if cause == "oom":     return "limit_bump"
        return "restart"  # cpu, network, unknown
    elif risk == "medium":
        return "limit_bump"
    return "alert_only"


def execute_action(action_type: str, pod: str, ns: str, dep: str, node: str, dry_run: bool):
    if dry_run:
        return f"DRY-RUN: {action_type} on {pod}", "dry_run"
    try:
        if MOCK_K8S:
            detail = {
                "scale_up":   lambda: mock_scale(ns, dep),
                "restart":    lambda: mock_restart(ns, dep),
                "cordon":     lambda: mock_cordon(node),
                "limit_bump": lambda: mock_bump(ns, dep),
            }.get(action_type, lambda: "Alert sent")()
        else:
            detail = {
                "scale_up":   lambda: real_scale_deployment(ns, dep),
                "restart":    lambda: real_rolling_restart(ns, dep),
                "cordon":     lambda: real_cordon_node(node),
                "limit_bump": lambda: real_bump_memory(ns, dep),
            }.get(action_type, lambda: "Alert sent")()
        return detail, "success"
    except Exception as e:
        prom_errors_total.labels(component="execute_action").inc()
        return str(e), "failed"


def remediate(pred: dict, rc: Optional[redis.Redis]) -> Optional[RemediationAction]:
    pod   = pred.get("pod", "")
    ns    = pred.get("namespace", "default")
    dep   = pred.get("deployment", "")
    node  = pred.get("node", "unknown")
    risk  = pred.get("risk_level", "low")
    score = pred.get("risk_score", 0)
    cause = pred.get("predicted_cause", "unknown")
    eta   = pred.get("eta_minutes")

    if risk == "low" or is_cooling_down(pod, rc):
        return None

    action_type = decide_action(pred)
    if not AUTO_REMEDIATE and action_type != "alert_only":
        action_type = "alert_only"

    log.info(f"Remediating {pod} [{risk}/{int(score * 100)}%] -> {action_type}")
    detail, result = execute_action(action_type, pod, ns, dep, node, DRY_RUN)
    if DRY_RUN:
        result = "dry_run"

    prom_actions_total.labels(action_type=action_type, result=result).inc()

    action = RemediationAction(
        pod=pod, namespace=ns, deployment=dep, node=node,
        action_type=action_type, risk_level=risk, risk_score=score,
        predicted_cause=cause, eta_minutes=eta, dry_run=DRY_RUN,
        result=result, detail=str(detail), timestamp=time.time(),
    )
    set_cooldown(pod, rc)
    store_action(action, rc)
    if risk in ("high", "critical"):
        notify_slack(action)
    return action


# ── OPERATOR LOOP ─────────────────────────────────────────────────
def operator_loop():
    log.info(
        f"Operator started | mock={MOCK_K8S} | auto={AUTO_REMEDIATE} "
        f"| dry_run={DRY_RUN} | slack={'yes' if SLACK_WEBHOOK else 'no'}"
    )
    retry_delay = 5
    while True:
        try:
            with prom_cycle_duration.time():
                r = requests.get(f"{PREDICTOR_URL}/predictions", timeout=10)
                r.raise_for_status()
                preds = r.json().get("predictions", [])
                prom_active_pods.set(len(preds))
                rc   = get_redis()
                done = 0
                for pred in preds:
                    if pred.get("risk_level", "low") != "low":
                        if remediate(pred, rc):
                            done += 1
                if preds:
                    log.info(f"Cycle: {len(preds)} pods, {done} actions taken")
            retry_delay = 5
        except requests.exceptions.ConnectionError:
            log.warning(f"Predictor unreachable, retry in {retry_delay}s")
            prom_errors_total.labels(component="predictor_connection").inc()
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
            continue
        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            prom_errors_total.labels(component="operator_loop").inc()
        time.sleep(LOOP_INTERVAL)


# ── AUTH ──────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        return True
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True


# ── FASTAPI APP ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    if not MOCK_K8S:
        init_k8s_client()
    threading.Thread(target=operator_loop, daemon=True).start()
    log.info("Operator API ready")
    yield
    log.info("Operator shutting down")


app = FastAPI(title="K8s Autopilot Operator", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    rc    = get_redis()
    stats = {}
    if rc:
        try:
            stats = rc.hgetall(STATS_KEY)
        except redis.RedisError as e:
            log.debug(f"Failed to get stats: {e}")
    return {
        "status": "ok",
        "mock_k8s": MOCK_K8S,
        "auto_remediate": AUTO_REMEDIATE,
        "dry_run": DRY_RUN,
        "slack_configured": bool(SLACK_WEBHOOK),
        "stats": stats,
    }


@app.get("/actions")
def get_actions(limit: int = 50):
    rc = get_redis()
    if not rc:
        return {"actions": [], "count": 0}
    try:
        raw = rc.lrange(ACTIONS_KEY, 0, limit - 1)
        return {"actions": [json.loads(a) for a in raw], "count": len(raw)}
    except redis.RedisError as e:
        return {"actions": [], "error": str(e)}


@app.get("/actions/summary")
def actions_summary():
    rc = get_redis()
    if not rc:
        return {"total": 0}
    try:
        raw      = rc.lrange(ACTIONS_KEY, 0, 999)
        acts     = [json.loads(a) for a in raw]
        by_type: dict  = {}
        by_result: dict = {}
        for a in acts:
            at = a.get("action_type", "?")
            ar = a.get("result", "?")
            by_type[at]   = by_type.get(at, 0) + 1
            by_result[ar] = by_result.get(ar, 0) + 1
        return {"total": len(acts), "by_action_type": by_type, "by_result": by_result}
    except redis.RedisError as e:
        return {"total": 0, "error": str(e)}


@app.delete("/actions/clear", dependencies=[Depends(verify_api_key)])
def clear_actions():
    rc = get_redis()
    if rc:
        try:
            rc.delete(ACTIONS_KEY)
            rc.delete(STATS_KEY)
        except redis.RedisError as e:
            log.warning(f"Failed to clear actions: {e}")
    return {"status": "cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")