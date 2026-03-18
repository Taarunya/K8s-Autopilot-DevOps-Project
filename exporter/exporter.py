"""
K8s Autopilot - Metrics Exporter
Simulates real Kubernetes pod metrics via Prometheus.
"""
import os
import time
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List
from prometheus_client import start_http_server, Gauge, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EXPORTER] %(message)s")
log = logging.getLogger(__name__)

NUM_PODS       = int(os.getenv("NUM_PODS", "30"))
INJECT_ANOMALY = os.getenv("INJECT_ANOMALIES", "true").lower() == "true"
# Increase ANOMALY_RATE for faster dev/demo feedback (default ~1 injection per 5 min per pod)
ANOMALY_RATE   = float(os.getenv("ANOMALY_RATE", "0.003"))
PORT           = 8000

LABELS = ["pod", "namespace", "node", "deployment"]

memory_rss = Gauge("pod_memory_rss_bytes",        "Pod RSS memory usage",       LABELS)
memory_ws  = Gauge("pod_memory_working_set_bytes", "Pod working set memory",     LABELS)
memory_lim = Gauge("pod_memory_limit_bytes",       "Pod memory limit",           LABELS)
cpu_usage  = Gauge("pod_cpu_usage_cores",          "Pod CPU cores used",         LABELS)
cpu_limit  = Gauge("pod_cpu_limit_cores",          "Pod CPU limit",              LABELS)
cpu_throt  = Gauge("pod_cpu_throttled_ratio",      "Pod CPU throttle ratio 0-1", LABELS)
net_rx     = Gauge("pod_network_receive_bytes",    "Pod network bytes received",  LABELS)
net_tx     = Gauge("pod_network_transmit_bytes",   "Pod network bytes sent",      LABELS)
net_err    = Gauge("pod_network_errors_total",     "Pod network errors",          LABELS)
pod_status = Gauge("pod_ready",                    "Pod ready status",            LABELS)
restarts   = Counter("pod_restart_total",          "Pod restart count",           LABELS)
oom_events = Counter("pod_oom_kill_total",         "Pod OOMKill events",          LABELS)

NAMESPACES  = ["production", "staging", "ml-workloads", "infra", "streaming"]
DEPLOYMENTS = [
    "payments-api", "user-service", "auth-service", "api-gateway",
    "ml-inference", "data-processor", "cache-worker", "event-consumer",
    "notification-svc", "analytics-engine", "search-api", "order-service",
]
NODES = [f"ip-10-0-{i}-{j}" for i in range(1, 4) for j in [10, 87, 142, 201]]


@dataclass
class PodState:
    name: str
    namespace: str
    node: str
    deployment: str
    mem_baseline: float = 200 * 1024**2
    mem_limit:    float = 512 * 1024**2
    cpu_baseline: float = 0.2
    cpu_limit:    float = 1.0
    anomaly_type:      str   = "none"
    anomaly_start:     float = 0.0
    anomaly_duration:  float = 0.0
    # Cumulative counts (state)
    restart_count:     int   = 0
    oom_count:         int   = 0
    # Previous tick snapshot — used to compute per-tick deltas for Counter.inc()
    prev_restart_count: int  = 0
    prev_oom_count:     int  = 0
    net_bytes_rx:  float = 0.0
    net_bytes_tx:  float = 0.0
    net_errors:    float = 0.0


def make_pods(n: int) -> List[PodState]:
    pods = []
    for i in range(n):
        dep    = random.choice(DEPLOYMENTS)
        ns     = random.choice(NAMESPACES)
        node   = random.choice(NODES)
        suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5))
        name   = f"{dep}-{random.randint(1000, 9999)}-{suffix}"
        mem_base  = random.uniform(50, 350) * 1024**2
        mem_limit = mem_base * random.uniform(2.0, 4.0)
        cpu_base  = random.uniform(0.05, 0.6)
        cpu_limit = cpu_base * random.uniform(2.0, 5.0)
        pods.append(PodState(
            name=name, namespace=ns, node=node, deployment=dep,
            mem_baseline=mem_base, mem_limit=mem_limit,
            cpu_baseline=cpu_base, cpu_limit=cpu_limit,
        ))
    return pods


def maybe_inject_anomaly(pod: PodState, now: float):
    if not INJECT_ANOMALY or pod.anomaly_type != "none":
        return
    if random.random() > ANOMALY_RATE:
        return
    pod.anomaly_type     = random.choice(["oom_leak", "cpu_spike", "net_storm"])
    pod.anomaly_start    = now
    pod.anomaly_duration = random.uniform(120, 600)
    log.info(f"Injecting {pod.anomaly_type} on {pod.name}")


def compute_metrics(pod: PodState, now: float, t: float) -> dict:
    hour_of_day  = (t / 3600) % 24
    daily_factor = 0.7 + 0.3 * math.sin(math.pi * hour_of_day / 12)
    mem  = pod.mem_baseline * daily_factor * random.uniform(0.95, 1.05)
    cpu  = pod.cpu_baseline * daily_factor * random.uniform(0.90, 1.10)
    rx   = random.uniform(1e5, 5e6) * daily_factor
    tx   = rx * random.uniform(0.3, 0.8)
    errs = random.uniform(0, 2)

    if pod.anomaly_type != "none":
        elapsed  = now - pod.anomaly_start
        progress = min(elapsed / pod.anomaly_duration, 1.0)

        if pod.anomaly_type == "oom_leak":
            mem = pod.mem_baseline * (1.0 + progress ** 2 * 3.5)
            if mem >= pod.mem_limit * 0.99:
                pod.oom_count     += 1
                pod.restart_count += 1
                pod.mem_baseline   = pod.mem_baseline * 0.6
                pod.anomaly_type   = "none"
                log.warning(f"OOMKill simulated on {pod.name}")
        elif pod.anomaly_type == "cpu_spike":
            cpu = min(pod.cpu_baseline + progress * pod.cpu_limit * 1.2, pod.cpu_limit)
        elif pod.anomaly_type == "net_storm":
            rx   = rx   * (1 + progress * 20)
            tx   = tx   * (1 + progress * 20)
            errs = errs * (1 + progress * 100)

        if elapsed >= pod.anomaly_duration:
            pod.anomaly_type = "none"

    pod.net_bytes_rx += rx
    pod.net_bytes_tx += tx
    pod.net_errors   += errs
    cpu_throttle = max(0, min(1, (cpu - pod.cpu_baseline) / (pod.cpu_limit - pod.cpu_baseline + 1e-9)))

    # Compute per-tick deltas so we can call Counter.inc() correctly
    restart_delta           = pod.restart_count - pod.prev_restart_count
    oom_delta               = pod.oom_count     - pod.prev_oom_count
    pod.prev_restart_count  = pod.restart_count
    pod.prev_oom_count      = pod.oom_count

    return {
        "mem_rss":       mem,
        "mem_ws":        mem * random.uniform(0.85, 1.0),
        "mem_limit":     pod.mem_limit,
        "cpu":           cpu,
        "cpu_limit":     pod.cpu_limit,
        "cpu_throt":     cpu_throttle,
        "net_rx":        pod.net_bytes_rx,
        "net_tx":        pod.net_bytes_tx,
        "net_err":       pod.net_errors,
        "ready":         1.0,
        "restart_delta": restart_delta,
        "oom_delta":     oom_delta,
    }


def update_prometheus(pod: PodState, m: dict):
    lv = [pod.name, pod.namespace, pod.node, pod.deployment]
    memory_rss.labels(*lv).set(m["mem_rss"])
    memory_ws.labels(*lv).set(m["mem_ws"])
    memory_lim.labels(*lv).set(m["mem_limit"])
    cpu_usage.labels(*lv).set(m["cpu"])
    cpu_limit.labels(*lv).set(m["cpu_limit"])
    cpu_throt.labels(*lv).set(m["cpu_throt"])
    net_rx.labels(*lv).set(m["net_rx"])
    net_tx.labels(*lv).set(m["net_tx"])
    net_err.labels(*lv).set(m["net_err"])
    pod_status.labels(*lv).set(m["ready"])
    # FIX: Counters must use .inc() — never .set(). Drive them from per-tick deltas.
    if m["restart_delta"] > 0:
        restarts.labels(*lv).inc(m["restart_delta"])
    if m["oom_delta"] > 0:
        oom_events.labels(*lv).inc(m["oom_delta"])


def main():
    log.info(f"Starting exporter on port {PORT} with {NUM_PODS} pods | anomaly_rate={ANOMALY_RATE}")
    start_http_server(PORT)
    pods  = make_pods(NUM_PODS)
    start = time.time()
    while True:
        now = time.time()
        t   = now - start
        for pod in pods:
            maybe_inject_anomaly(pod, now)
            m = compute_metrics(pod, now, t)
            update_prometheus(pod, m)
        time.sleep(15)


if __name__ == "__main__":
    main()