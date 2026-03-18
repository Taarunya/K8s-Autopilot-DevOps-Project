"""
Unit tests for the K8s Autopilot Predictor service.
Tests feature extraction, training, prediction, and API endpoints.
"""
import sys
import os
import time
import pytest
import numpy as np

# Add predictor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictor"))

from predictor import (
    AutopilotPredictor,
    PodMetricWindow,
    PodPrediction,
    extract_features,
    determine_cause,
    estimate_eta,
    RISK_HIGH,
    RISK_MEDIUM,
)


# ── FIXTURES ──────────────────────────────────────────────────────
@pytest.fixture
def predictor_instance():
    """Create a trained predictor."""
    p = AutopilotPredictor()
    data = p.generate_synthetic_training_data(n_normal=200, n_failure=50)
    p.train(data)
    return p


@pytest.fixture
def normal_window():
    """Pod with normal, healthy metrics."""
    n = 20
    return PodMetricWindow(
        pod="user-service-1234-abc",
        namespace="production",
        node="ip-10-0-1-10",
        deployment="user-service",
        timestamps=[time.time() - (n - i) * 15 for i in range(n)],
        mem_rss=[200 * 1024**2 + np.random.normal(0, 5e6) for _ in range(n)],
        mem_limit=[512 * 1024**2] * n,
        cpu_usage=[0.2 + np.random.normal(0, 0.02) for _ in range(n)],
        cpu_limit=[1.0] * n,
        cpu_throttle=[0.05 + np.random.normal(0, 0.01) for _ in range(n)],
        net_rx=[1e6 + np.random.normal(0, 1e5) for _ in range(n)],
        net_errors=[0.5 + np.random.normal(0, 0.1) for _ in range(n)],
        restart_count=[0.0] * n,
        oom_count=[0.0] * n,
    )


@pytest.fixture
def oom_window():
    """Pod with OOM leak pattern — rising memory."""
    n = 20
    base_mem = 400 * 1024**2
    return PodMetricWindow(
        pod="payments-api-5678-xyz",
        namespace="production",
        node="ip-10-0-2-87",
        deployment="payments-api",
        timestamps=[time.time() - (n - i) * 15 for i in range(n)],
        mem_rss=[base_mem + i * 6 * 1024**2 for i in range(n)],  # steadily rising
        mem_limit=[512 * 1024**2] * n,
        cpu_usage=[0.3] * n,
        cpu_limit=[1.0] * n,
        cpu_throttle=[0.1] * n,
        net_rx=[1e6] * n,
        net_errors=[0.5] * n,
        restart_count=[0.0] * (n - 2) + [1.0, 2.0],
        oom_count=[0.0] * (n - 2) + [1.0, 2.0],
    )


@pytest.fixture
def cpu_spike_window():
    """Pod with CPU spike — high throttle."""
    n = 20
    return PodMetricWindow(
        pod="api-gateway-9999-def",
        namespace="production",
        node="ip-10-0-3-142",
        deployment="api-gateway",
        timestamps=[time.time() - (n - i) * 15 for i in range(n)],
        mem_rss=[200 * 1024**2] * n,
        mem_limit=[512 * 1024**2] * n,
        cpu_usage=[0.3 + i * 0.04 for i in range(n)],  # rising CPU
        cpu_limit=[1.0] * n,
        cpu_throttle=[0.2 + i * 0.04 for i in range(n)],  # rising throttle
        net_rx=[1e6] * n,
        net_errors=[0.5] * n,
        restart_count=[0.0] * n,
        oom_count=[0.0] * n,
    )


@pytest.fixture
def network_storm_window():
    """Pod with network storm — high error rate and rx spike."""
    n = 20
    return PodMetricWindow(
        pod="event-consumer-4444-ghi",
        namespace="streaming",
        node="ip-10-0-1-201",
        deployment="event-consumer",
        timestamps=[time.time() - (n - i) * 15 for i in range(n)],
        mem_rss=[200 * 1024**2] * n,
        mem_limit=[512 * 1024**2] * n,
        cpu_usage=[0.3] * n,
        cpu_limit=[1.0] * n,
        cpu_throttle=[0.05] * n,
        net_rx=[1e6 * (1 + i * 2) for i in range(n)],  # massive rx spike
        net_errors=[i * 5.0 for i in range(n)],  # rising errors
        restart_count=[0.0] * n,
        oom_count=[0.0] * n,
    )


# ── FEATURE EXTRACTION TESTS ─────────────────────────────────────
class TestFeatureExtraction:
    def test_normal_pod_features(self, normal_window):
        features = extract_features(normal_window)
        assert features is not None
        assert "mem_utilization" in features
        assert "cpu_utilization" in features
        assert "net_error_rate" in features
        assert "anomaly_score" in features
        # Normal pod should have moderate memory utilization
        assert 0.0 < features["mem_utilization"] < 0.8
        # Normal pod should have low restart rate
        assert features["restart_rate"] == 0.0
        assert features["oom_recent"] == 0.0

    def test_oom_pod_features(self, oom_window):
        features = extract_features(oom_window)
        assert features is not None
        # OOM pod should show high memory utilization
        assert features["mem_utilization"] > 0.5
        # OOM pod should have positive memory slope
        assert features["mem_slope"] > 0
        # Should have recent OOM events
        assert features["oom_recent"] > 0

    def test_cpu_spike_features(self, cpu_spike_window):
        features = extract_features(cpu_spike_window)
        assert features is not None
        # CPU spike should show rising CPU
        assert features["cpu_slope"] > 0
        # Should have elevated throttle
        assert features["cpu_throttle_ratio"] > 0.1

    def test_feature_keys(self, normal_window):
        features = extract_features(normal_window)
        expected_keys = [
            "mem_utilization", "mem_utilization_p99", "mem_slope", "mem_acceleration",
            "cpu_utilization", "cpu_throttle_ratio", "cpu_slope",
            "net_error_rate", "net_rx_spike",
            "restart_rate", "oom_recent", "anomaly_score",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_too_few_datapoints_returns_none(self):
        window = PodMetricWindow(
            pod="short-pod", namespace="test", node="n1", deployment="d1",
            timestamps=[1.0, 2.0], mem_rss=[100.0, 200.0],
            mem_limit=[500.0, 500.0], cpu_usage=[0.1, 0.2],
            cpu_limit=[1.0, 1.0], cpu_throttle=[0.0, 0.0],
            net_rx=[100.0, 200.0], net_errors=[0.0, 0.0],
            restart_count=[0.0, 0.0], oom_count=[0.0, 0.0],
        )
        result = extract_features(window)
        assert result is None  # less than 4 data points

    def test_net_rx_spike_capped(self, network_storm_window):
        features = extract_features(network_storm_window)
        assert features is not None
        assert features["net_rx_spike"] <= 50.0

    def test_restart_rate_capped(self):
        n = 20
        window = PodMetricWindow(
            pod="restart-pod", namespace="test", node="n1", deployment="d1",
            timestamps=[time.time() - (n - i) * 15 for i in range(n)],
            mem_rss=[200e6] * n, mem_limit=[500e6] * n,
            cpu_usage=[0.1] * n, cpu_limit=[1.0] * n, cpu_throttle=[0.0] * n,
            net_rx=[1e6] * n, net_errors=[0.0] * n,
            restart_count=[float(i * 100) for i in range(n)],
            oom_count=[0.0] * n,
        )
        features = extract_features(window)
        assert features is not None
        assert features["restart_rate"] <= 20.0


# ── CAUSE DETERMINATION TESTS ────────────────────────────────────
class TestDetermineCause:
    def test_oom_cause(self):
        features = {
            "mem_slope": 0.05, "mem_utilization_p99": 0.9, "oom_recent": 2.0,
            "cpu_throttle_ratio": 0.1, "cpu_slope": 0.001, "cpu_utilization": 0.3,
            "net_error_rate": 0.001, "net_rx_spike": 1.0,
        }
        assert determine_cause(features) == "oom"

    def test_cpu_cause(self):
        features = {
            "mem_slope": 0.0, "mem_utilization_p99": 0.3, "oom_recent": 0.0,
            "cpu_throttle_ratio": 0.9, "cpu_slope": 0.05, "cpu_utilization": 0.95,
            "net_error_rate": 0.001, "net_rx_spike": 1.0,
        }
        assert determine_cause(features) == "cpu"

    def test_network_cause(self):
        features = {
            "mem_slope": 0.0, "mem_utilization_p99": 0.3, "oom_recent": 0.0,
            "cpu_throttle_ratio": 0.05, "cpu_slope": 0.001, "cpu_utilization": 0.2,
            "net_error_rate": 0.3, "net_rx_spike": 15.0,
        }
        assert determine_cause(features) == "network"

    def test_unknown_cause(self):
        features = {
            "mem_slope": 0.0, "mem_utilization_p99": 0.0, "oom_recent": 0.0,
            "cpu_throttle_ratio": 0.0, "cpu_slope": 0.0, "cpu_utilization": 0.0,
            "net_error_rate": 0.0, "net_rx_spike": 0.0,
        }
        assert determine_cause(features) == "unknown"


# ── ETA ESTIMATION TESTS ─────────────────────────────────────────
class TestEstimateEta:
    def test_low_risk_returns_none(self):
        features = {"mem_slope": 0.01, "mem_utilization": 0.5}
        assert estimate_eta(features, 0.3) is None

    def test_high_risk_with_memory_slope(self):
        features = {"mem_slope": 0.01, "mem_utilization": 0.7}
        eta = estimate_eta(features, 0.8)
        assert eta is not None
        assert 1 <= eta <= 120

    def test_critical_risk_without_slope(self):
        features = {"mem_slope": 0.0, "mem_utilization": 0.9}
        eta = estimate_eta(features, 0.9)
        assert eta is not None
        assert eta >= 1

    def test_eta_bounded(self):
        features = {"mem_slope": 0.0001, "mem_utilization": 0.1}
        eta = estimate_eta(features, 0.8)
        if eta is not None:
            assert 1 <= eta <= 120


# ── MODEL TRAINING TESTS ─────────────────────────────────────────
class TestModelTraining:
    def test_training_with_enough_data(self):
        p = AutopilotPredictor()
        data = p.generate_synthetic_training_data(n_normal=200, n_failure=50)
        metrics = p.train(data)
        assert p.is_trained is True
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_training_with_insufficient_data(self):
        p = AutopilotPredictor()
        data = [({"mem_utilization": 0.5}, 0)] * 5
        metrics = p.train(data)
        assert metrics.get("status") == "skipped"
        assert p.is_trained is False

    def test_synthetic_data_distribution(self):
        p = AutopilotPredictor()
        data = p.generate_synthetic_training_data(n_normal=100, n_failure=25)
        assert len(data) == 125
        labels = [label for _, label in data]
        assert labels.count(0) == 100
        assert labels.count(1) == 25

    def test_model_performance_above_threshold(self):
        """The model should achieve reasonable F1 on synthetic data."""
        p = AutopilotPredictor()
        data = p.generate_synthetic_training_data(n_normal=600, n_failure=150)
        metrics = p.train(data)
        # With clear synthetic signal, F1 should be reasonable
        assert metrics["f1"] > 0.5, f"F1 too low: {metrics['f1']}"


# ── PREDICTION TESTS ─────────────────────────────────────────────
class TestPrediction:
    def test_predict_normal_pod(self, predictor_instance, normal_window):
        pred = predictor_instance.predict(normal_window)
        assert pred is not None
        assert isinstance(pred, PodPrediction)
        assert pred.pod == "user-service-1234-abc"
        assert pred.namespace == "production"
        assert 0.0 <= pred.risk_score <= 1.0
        assert pred.risk_level in ("low", "medium", "high", "critical")

    def test_predict_oom_pod(self, predictor_instance, oom_window):
        pred = predictor_instance.predict(oom_window)
        assert pred is not None
        # OOM pod should have elevated risk
        assert pred.risk_score > 0.0
        assert pred.predicted_cause in ("oom", "cpu", "network", "unknown")

    def test_predict_cpu_spike(self, predictor_instance, cpu_spike_window):
        pred = predictor_instance.predict(cpu_spike_window)
        assert pred is not None
        assert pred.risk_score >= 0.0

    def test_prediction_fields(self, predictor_instance, normal_window):
        pred = predictor_instance.predict(normal_window)
        assert pred is not None
        assert pred.pod != ""
        assert pred.namespace != ""
        assert pred.deployment != ""
        assert pred.timestamp > 0
        assert 0.0 <= pred.confidence <= 1.0
        assert isinstance(pred.features, dict)

    def test_risk_levels_mapping(self, predictor_instance):
        """Verify risk level thresholds are logical."""
        # A heuristic check: higher scores → higher risk levels
        assert RISK_MEDIUM < RISK_HIGH

    def test_untrained_predictor_uses_heuristic(self, normal_window):
        p = AutopilotPredictor()
        # Not trained yet
        p.is_trained = False
        p.model = None
        pred = p.predict(normal_window)
        assert pred is not None
        assert 0.0 <= pred.risk_score <= 1.0


# ── HEURISTIC FALLBACK TESTS ─────────────────────────────────────
class TestHeuristic:
    def test_heuristic_normal(self):
        p = AutopilotPredictor()
        features = {
            "mem_utilization_p99": 0.3, "mem_slope": 0.001,
            "cpu_throttle_ratio": 0.05, "net_error_rate": 0.001,
            "restart_rate": 0.0, "oom_recent": 0.0,
        }
        score = p._heuristic(features)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # normal pod should be low risk

    def test_heuristic_high_memory(self):
        p = AutopilotPredictor()
        features = {
            "mem_utilization_p99": 0.95, "mem_slope": 0.05,
            "cpu_throttle_ratio": 0.1, "net_error_rate": 0.001,
            "restart_rate": 0.0, "oom_recent": 2.0,
        }
        score = p._heuristic(features)
        assert score > 0.5  # high memory should flag

    def test_heuristic_bounded(self):
        p = AutopilotPredictor()
        features = {
            "mem_utilization_p99": 1.0, "mem_slope": 1.0,
            "cpu_throttle_ratio": 1.0, "net_error_rate": 1.0,
            "restart_rate": 100.0, "oom_recent": 100.0,
        }
        score = p._heuristic(features)
        assert score <= 1.0
