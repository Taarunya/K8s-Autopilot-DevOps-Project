"""
K8s Autopilot - Prometheus Data Fetcher
"""
import os
import time
import logging
import requests
from typing import List, Dict
from predictor import PodMetricWindow

log = logging.getLogger(__name__)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", "30"))
STEP_SECONDS   = 15


def query_range(metric: str) -> Dict:
    end   = int(time.time())
    start = end - WINDOW_MINUTES * 60
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": metric, "start": start, "end": end, "step": STEP_SECONDS},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.error(f"Prometheus query failed [{metric}]: {e}")
        return {"data": {"result": []}}


def parse_series(result: Dict) -> Dict:
    pod_series = {}
    for series in result.get("data", {}).get("result", []):
        labels = series.get("metric", {})
        pod    = labels.get("pod", "unknown")
        values = series.get("values", [])
        if not values:
            continue
        pod_series[pod] = {
            "timestamps": [float(v[0]) for v in values],
            "values":     [float(v[1]) for v in values],
            "labels":     labels,
        }
    return pod_series


def fetch_pod_windows() -> List[PodMetricWindow]:
    log.debug("Fetching pod metric windows...")

    raw = {
        "mem_rss":   query_range("pod_memory_rss_bytes"),
        "mem_limit": query_range("pod_memory_limit_bytes"),
        "cpu":       query_range("pod_cpu_usage_cores"),
        "cpu_limit": query_range("pod_cpu_limit_cores"),
        "cpu_throt": query_range("pod_cpu_throttled_ratio"),
        "net_rx":    query_range("pod_network_receive_bytes"),
        "net_err":   query_range("pod_network_errors_total"),
        "restarts":  query_range("pod_restart_total"),
        "oom":       query_range("pod_oom_kill_total"),
    }

    parsed  = {k: parse_series(v) for k, v in raw.items()}
    all_pods = set(parsed["mem_rss"].keys())
    windows = []

    for pod in all_pods:
        mem_data = parsed["mem_rss"].get(pod, {})
        if not mem_data:
            continue
        labels = mem_data.get("labels", {})
        ts     = mem_data["timestamps"]

        def get_vals(key):
            s = parsed[key].get(pod, {})
            return s["values"] if s else [0.0] * len(ts)

        windows.append(PodMetricWindow(
            pod=pod,
            namespace=labels.get("namespace", "default"),
            node=labels.get("node", "unknown"),
            deployment=labels.get("deployment", pod.rsplit("-", 2)[0] if pod.count("-") >= 2 else pod),
            timestamps=ts,
            mem_rss=mem_data["values"],
            mem_limit=get_vals("mem_limit"),
            cpu_usage=get_vals("cpu"),
            cpu_limit=get_vals("cpu_limit"),
            cpu_throttle=get_vals("cpu_throt"),
            net_rx=get_vals("net_rx"),
            net_errors=get_vals("net_err"),
            restart_count=get_vals("restarts"),
            oom_count=get_vals("oom"),
        ))

    log.info(f"Fetched windows for {len(windows)} pods")
    return windows