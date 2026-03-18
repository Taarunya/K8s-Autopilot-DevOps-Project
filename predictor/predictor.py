import os
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

log = logging.getLogger(__name__)

MODEL_DIR   = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RISK_HIGH   = float(os.getenv("RISK_THRESHOLD_HIGH",   "0.75"))
RISK_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.50"))

FEATURE_COLS = [
    "mem_utilization", "mem_utilization_p99", "mem_slope", "mem_acceleration",
    "cpu_utilization", "cpu_throttle_ratio",  "cpu_slope",
    "net_error_rate",  "net_rx_spike",
    "restart_rate",    "oom_recent", "anomaly_score",
]

ISO_COLS = [c for c in FEATURE_COLS if c != "anomaly_score"]


@dataclass
class PodMetricWindow:
    pod: str
    namespace: str
    node: str
    deployment: str
    timestamps:    List[float]
    mem_rss:       List[float]
    mem_limit:     List[float]
    cpu_usage:     List[float]
    cpu_limit:     List[float]
    cpu_throttle:  List[float]
    net_rx:        List[float]
    net_errors:    List[float]
    restart_count: List[float]
    oom_count:     List[float]


@dataclass
class PodPrediction:
    pod: str
    namespace: str
    node: str
    deployment: str
    risk_score:      float
    risk_level:      str
    predicted_cause: str
    eta_minutes:     Optional[int]
    confidence:      float
    features:        Dict
    timestamp:       float


def extract_features(w: PodMetricWindow) -> Optional[Dict]:
    try:
        rss  = np.array(w.mem_rss,       dtype=float)
        lim  = np.array(w.mem_limit,     dtype=float)
        cpu  = np.array(w.cpu_usage,     dtype=float)
        clim = np.array(w.cpu_limit,     dtype=float)
        thrt = np.array(w.cpu_throttle,  dtype=float)
        rx   = np.array(w.net_rx,        dtype=float)
        errs = np.array(w.net_errors,    dtype=float)
        rst  = np.array(w.restart_count, dtype=float)
        oom  = np.array(w.oom_count,     dtype=float)

        if len(rss) < 4:
            return None

        util     = rss / (lim + 1e-9)
        x        = np.arange(len(rss))
        mem_util = float(util.mean())
        mem_p99  = float(np.percentile(util, 99))
        mem_slp  = float(np.polyfit(x, util, 1)[0]) if len(rss) > 1 else 0.0
        half     = len(rss) // 2
        s1 = float(np.polyfit(x[:half], util[:half], 1)[0]) if half > 1 else 0.0
        s2 = float(np.polyfit(x[half:], util[half:], 1)[0]) if half > 1 else 0.0
        mem_acc  = s2 - s1

        cpu_util  = float((cpu / (clim + 1e-9)).mean())
        cpu_thm   = float(thrt.mean())
        cpu_slp   = float(np.polyfit(x, cpu / (clim + 1e-9), 1)[0]) if len(cpu) > 1 else 0.0

        rx_mean   = rx.mean() if rx.mean() > 0 else 1.0
        rx_spike  = float(rx[-1] / rx_mean) if len(rx) > 0 else 1.0
        err_rate  = float(errs[-1] / (rx[-1] + 1e-9)) if len(errs) > 0 else 0.0

        hrs = max((w.timestamps[-1] - w.timestamps[0]) / 3600, 0.01)
        rst_rate  = float((rst[-1] - rst[0]) / hrs) if len(rst) > 1 else 0.0
        oom_rec   = float(oom[-1] - oom[0])          if len(oom) > 1 else 0.0

        return {
            "mem_utilization":     mem_util,
            "mem_utilization_p99": mem_p99,
            "mem_slope":           mem_slp,
            "mem_acceleration":    mem_acc,
            "cpu_utilization":     cpu_util,
            "cpu_throttle_ratio":  cpu_thm,
            "cpu_slope":           cpu_slp,
            "net_error_rate":      err_rate,
            "net_rx_spike":        min(rx_spike, 50.0),
            "restart_rate":        min(rst_rate, 20.0),
            "oom_recent":          oom_rec,
            "anomaly_score":       0.0,  # filled in during predict()
        }
    except Exception as e:
        log.warning(f"Feature extraction failed for {w.pod}: {e}")
        return None


def determine_cause(f: Dict) -> str:
    scores = {
        "oom":     f["mem_slope"] * 10 + f["mem_utilization_p99"] + f["oom_recent"],
        "cpu":     f["cpu_throttle_ratio"] + f["cpu_slope"] * 5 + f["cpu_utilization"],
        "network": f["net_error_rate"] * 10 + max(0, f["net_rx_spike"] - 2),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.1 else "unknown"


def estimate_eta(f: Dict, risk: float) -> Optional[int]:
    if risk < RISK_MEDIUM:
        return None
    if f["mem_slope"] > 0.001:
        headroom = max(0, 1.0 - f["mem_utilization"])
        eta      = int(headroom / (f["mem_slope"] + 1e-9) * 0.25)
        return max(1, min(eta, 120))
    if risk >= RISK_HIGH:
        return int((1.0 - risk) * 60)
    return None


class AutopilotPredictor:
    def __init__(self):
        self.model      = None
        self.iso_forest = None
        self.is_trained = False
        self.metrics    = {}
        self._load_or_init()

    def _model_path(self): return MODEL_DIR / "autopilot_model.joblib"
    def _iso_path(self):   return MODEL_DIR / "iso_forest.joblib"

    def _load_or_init(self):
        if self._model_path().exists():
            try:
                self.model      = joblib.load(self._model_path())
                self.iso_forest = joblib.load(self._iso_path())
                self.is_trained = True
                log.info("Loaded existing model from disk")
                return
            except Exception as e:
                log.warning(f"Could not load saved model: {e} — reinitialising")
        self._init_models()

    def _init_models(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08,
                max_depth=4, min_samples_leaf=5,
                subsample=0.85, random_state=42,
            ))
        ])
        self.iso_forest = IsolationForest(
            n_estimators=100, contamination=0.1, random_state=42
        )

    def train(self, training_data: List[Tuple[Dict, int]]) -> Dict:
        if len(training_data) < 20:
            return {"status": "skipped", "reason": "not enough data"}

        X_raw = pd.DataFrame([f for f, _ in training_data])[ISO_COLS].fillna(0)
        y     = np.array([l for _, l in training_data])

        # Step 1: fit IsolationForest on base features (no anomaly_score column)
        self.iso_forest.fit(X_raw)
        iso_scores = -self.iso_forest.score_samples(X_raw)

        # Step 2: build full feature matrix with anomaly_score filled in
        X_full = X_raw.copy()
        X_full["anomaly_score"] = iso_scores
        X_full = X_full[FEATURE_COLS]

        # Step 3: train GBM classifier
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        self.is_trained = True

        y_pred = self.model.predict(X_te)
        self.metrics = {
            "f1":         round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
            "precision":  round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
            "samples":    len(training_data),
            "trained_at": time.time(),
        }

        try:
            joblib.dump(self.model,      self._model_path())
            joblib.dump(self.iso_forest, self._iso_path())
        except Exception as e:
            log.warning(f"Could not save model: {e}")

        log.info(f"Model trained: {self.metrics}")
        return self.metrics

    def predict(self, window: PodMetricWindow) -> Optional[PodPrediction]:
        features = extract_features(window)
        if features is None:
            return None

        # Compute anomaly score from ISO_COLS only
        try:
            if self.is_trained and self.iso_forest:
                X_iso = pd.DataFrame([features])[ISO_COLS].fillna(0)
                features["anomaly_score"] = float(
                    -self.iso_forest.score_samples(X_iso)[0]
                )
            else:
                features["anomaly_score"] = features.get("mem_utilization_p99", 0.0)
        except Exception as e:
            log.warning(f"IsoForest scoring failed for {window.pod}: {e}")
            features["anomaly_score"] = features.get("mem_utilization_p99", 0.0)

        # Predict risk score
        if self.is_trained and self.model:
            try:
                X          = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
                proba      = self.model.predict_proba(X)[0]
                risk_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                # FIX: use real model confidence (max class probability) instead of stub formula
                confidence = round(float(max(proba)), 4)
            except Exception as e:
                log.warning(f"GBM prediction failed for {window.pod}: {e}")
                risk_score = self._heuristic(features)
                confidence = round(min(0.99, risk_score + 0.05) if risk_score > 0.5 else max(0.5, 1 - risk_score), 4)
        else:
            risk_score = self._heuristic(features)
            confidence = round(min(0.99, risk_score + 0.05) if risk_score > 0.5 else max(0.5, 1 - risk_score), 4)

        if   risk_score >= 0.85:       risk_level = "critical"
        elif risk_score >= RISK_HIGH:  risk_level = "high"
        elif risk_score >= RISK_MEDIUM:risk_level = "medium"
        else:                          risk_level = "low"

        return PodPrediction(
            pod=window.pod,
            namespace=window.namespace,
            node=window.node,
            deployment=window.deployment,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            predicted_cause=determine_cause(features),
            eta_minutes=estimate_eta(features, risk_score),
            confidence=confidence,
            features={k: round(float(v), 6) for k, v in features.items()},
            timestamp=time.time(),
        )

    def _heuristic(self, f: Dict) -> float:
        s  = f.get("mem_utilization_p99", 0) * 0.4
        s += min(f.get("mem_slope", 0) * 20, 0.3)
        s += f.get("cpu_throttle_ratio", 0) * 0.15
        s += min(f.get("net_error_rate", 0) * 5, 0.1)
        s += min(f.get("restart_rate", 0) * 0.05, 0.1)
        s += min(f.get("oom_recent", 0) * 0.2, 0.2)
        return min(float(s), 1.0)

    def generate_synthetic_training_data(self, n_normal=400, n_failure=100):
        data = []

        for _ in range(n_normal):
            f = {
                "mem_utilization":     np.random.uniform(0.1, 0.6),
                "mem_utilization_p99": np.random.uniform(0.15, 0.65),
                "mem_slope":           np.random.uniform(-0.001, 0.003),
                "mem_acceleration":    np.random.uniform(-0.001, 0.001),
                "cpu_utilization":     np.random.uniform(0.05, 0.5),
                "cpu_throttle_ratio":  np.random.uniform(0.0, 0.15),
                "cpu_slope":           np.random.uniform(-0.002, 0.002),
                "net_error_rate":      np.random.uniform(0, 0.005),
                "net_rx_spike":        np.random.uniform(0.8, 2.0),
                "restart_rate":        np.random.uniform(0, 0.5),
                "oom_recent":          0.0,
                # FIX: anomaly_score for normal samples should be genuinely low
                "anomaly_score":       np.random.uniform(0, 0.3),
            }
            data.append((f, 0))

        for _ in range(n_failure):
            cause = np.random.choice(["oom", "cpu", "net"])
            # FIX: start with clearly failure-range anomaly_score, then override cause features
            # This ensures clean separation between classes for the GBM
            if cause == "oom":
                f = {
                    "mem_utilization":     np.random.uniform(0.75, 0.99),
                    "mem_utilization_p99": np.random.uniform(0.85, 0.99),
                    "mem_slope":           np.random.uniform(0.01, 0.08),
                    "mem_acceleration":    np.random.uniform(0.005, 0.03),
                    "cpu_utilization":     np.random.uniform(0.05, 0.5),
                    "cpu_throttle_ratio":  np.random.uniform(0.0, 0.2),
                    "cpu_slope":           np.random.uniform(-0.002, 0.002),
                    "net_error_rate":      np.random.uniform(0, 0.005),
                    "net_rx_spike":        np.random.uniform(0.8, 2.0),
                    "restart_rate":        np.random.uniform(0, 1.0),
                    "oom_recent":          float(np.random.choice([0, 1, 2])),
                    "anomaly_score":       np.random.uniform(0.6, 1.0),
                }
            elif cause == "cpu":
                f = {
                    "mem_utilization":     np.random.uniform(0.1, 0.5),
                    "mem_utilization_p99": np.random.uniform(0.1, 0.55),
                    "mem_slope":           np.random.uniform(-0.001, 0.003),
                    "mem_acceleration":    np.random.uniform(-0.001, 0.001),
                    "cpu_utilization":     np.random.uniform(0.8, 1.2),
                    "cpu_throttle_ratio":  np.random.uniform(0.6, 1.0),
                    "cpu_slope":           np.random.uniform(0.02, 0.1),
                    "net_error_rate":      np.random.uniform(0, 0.005),
                    "net_rx_spike":        np.random.uniform(0.8, 2.0),
                    "restart_rate":        np.random.uniform(0, 0.5),
                    "oom_recent":          0.0,
                    "anomaly_score":       np.random.uniform(0.55, 1.0),
                }
            else:  # net
                f = {
                    "mem_utilization":     np.random.uniform(0.1, 0.5),
                    "mem_utilization_p99": np.random.uniform(0.1, 0.5),
                    "mem_slope":           np.random.uniform(-0.001, 0.003),
                    "mem_acceleration":    np.random.uniform(-0.001, 0.001),
                    "cpu_utilization":     np.random.uniform(0.05, 0.5),
                    "cpu_throttle_ratio":  np.random.uniform(0.0, 0.15),
                    "cpu_slope":           np.random.uniform(-0.002, 0.002),
                    "net_error_rate":      np.random.uniform(0.05, 0.5),
                    "net_rx_spike":        np.random.uniform(5, 30),
                    "restart_rate":        np.random.uniform(0, 0.5),
                    "oom_recent":          0.0,
                    "anomaly_score":       np.random.uniform(0.5, 1.0),
                }
            data.append((f, 1))

        np.random.shuffle(data)
        return data