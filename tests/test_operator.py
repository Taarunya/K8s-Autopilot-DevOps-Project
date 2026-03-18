"""
Unit tests for the K8s Autopilot Operator service.
Tests decision logic, remediation flow, cooldowns, and API endpoints.
"""
import sys
import os
import json
import time
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# Add operator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "operator"))

# Set env vars before import
os.environ["MOCK_K8S"] = "true"
os.environ["AUTO_REMEDIATE"] = "true"
os.environ["DRY_RUN"] = "false"
os.environ["SLACK_WEBHOOK_URL"] = ""
os.environ["API_KEY"] = "test-api-key"

from app import (
    decide_action,
    execute_action,
    remediate,
    RemediationAction,
    mock_scale,
    mock_restart,
    mock_cordon,
    mock_bump,
    is_cooling_down,
    set_cooldown,
    store_action,
)


# ── DECISION LOGIC TESTS ─────────────────────────────────────────
class TestDecideAction:
    def test_critical_oom_scales_up(self):
        pred = {"risk_level": "critical", "predicted_cause": "oom"}
        assert decide_action(pred) == "scale_up"

    def test_critical_cpu_cordons(self):
        pred = {"risk_level": "critical", "predicted_cause": "cpu"}
        assert decide_action(pred) == "cordon"

    def test_critical_network_restarts(self):
        pred = {"risk_level": "critical", "predicted_cause": "network"}
        assert decide_action(pred) == "restart"

    def test_critical_unknown_restarts(self):
        pred = {"risk_level": "critical", "predicted_cause": "unknown"}
        assert decide_action(pred) == "restart"

    def test_high_oom_bumps_limits(self):
        pred = {"risk_level": "high", "predicted_cause": "oom"}
        assert decide_action(pred) == "limit_bump"

    def test_high_cpu_restarts(self):
        pred = {"risk_level": "high", "predicted_cause": "cpu"}
        assert decide_action(pred) == "restart"

    def test_high_network_restarts(self):
        pred = {"risk_level": "high", "predicted_cause": "network"}
        assert decide_action(pred) == "restart"

    def test_medium_any_bumps_limits(self):
        for cause in ["oom", "cpu", "network", "unknown"]:
            pred = {"risk_level": "medium", "predicted_cause": cause}
            assert decide_action(pred) == "limit_bump", f"Failed for cause={cause}"

    def test_low_alert_only(self):
        pred = {"risk_level": "low", "predicted_cause": "oom"}
        assert decide_action(pred) == "alert_only"

    def test_missing_fields_defaults_to_alert(self):
        pred = {}
        assert decide_action(pred) == "alert_only"


# ── MOCK ACTIONS TESTS ───────────────────────────────────────────
class TestMockActions:
    def test_mock_scale(self):
        result = mock_scale("production", "payments-api")
        assert "[MOCK]" in result
        assert "Scale" in result
        assert "production/payments-api" in result

    def test_mock_restart(self):
        result = mock_restart("staging", "user-service")
        assert "[MOCK]" in result
        assert "Restart" in result

    def test_mock_cordon(self):
        result = mock_cordon("ip-10-0-1-10")
        assert "[MOCK]" in result
        assert "Cordon" in result
        assert "ip-10-0-1-10" in result

    def test_mock_bump(self):
        result = mock_bump("production", "api-gateway")
        assert "[MOCK]" in result
        assert "Bump" in result


# ── EXECUTE ACTION TESTS ─────────────────────────────────────────
class TestExecuteAction:
    def test_dry_run(self):
        detail, result = execute_action("scale_up", "pod1", "ns1", "dep1", "n1", dry_run=True)
        assert result == "dry_run"
        assert "DRY-RUN" in detail

    def test_mock_scale_up(self):
        detail, result = execute_action("scale_up", "pod1", "production", "dep1", "n1", dry_run=False)
        assert result == "success"
        assert "[MOCK]" in detail

    def test_mock_restart(self):
        detail, result = execute_action("restart", "pod1", "production", "dep1", "n1", dry_run=False)
        assert result == "success"

    def test_mock_cordon(self):
        detail, result = execute_action("cordon", "pod1", "ns1", "dep1", "node1", dry_run=False)
        assert result == "success"

    def test_mock_limit_bump(self):
        detail, result = execute_action("limit_bump", "pod1", "ns1", "dep1", "n1", dry_run=False)
        assert result == "success"

    def test_alert_only(self):
        detail, result = execute_action("alert_only", "pod1", "ns1", "dep1", "n1", dry_run=False)
        assert result == "success"
        assert detail == "Alert sent"


# ── REMEDIATION FLOW TESTS ───────────────────────────────────────
class TestRemediation:
    def _make_pred(self, risk_level="high", cause="oom", score=0.85):
        return {
            "pod": "test-pod-1234-abc",
            "namespace": "production",
            "deployment": "test-service",
            "node": "ip-10-0-1-10",
            "risk_level": risk_level,
            "risk_score": score,
            "predicted_cause": cause,
            "eta_minutes": 15,
        }

    def test_low_risk_no_action(self):
        pred = self._make_pred(risk_level="low", score=0.2)
        result = remediate(pred, None)
        assert result is None

    def test_high_risk_takes_action(self):
        pred = self._make_pred(risk_level="high", cause="oom", score=0.85)
        action = remediate(pred, None)
        assert action is not None
        assert isinstance(action, RemediationAction)
        assert action.action_type == "limit_bump"
        assert action.risk_level == "high"
        assert action.pod == "test-pod-1234-abc"

    def test_critical_risk_takes_action(self):
        pred = self._make_pred(risk_level="critical", cause="cpu", score=0.95)
        action = remediate(pred, None)
        assert action is not None
        assert action.action_type == "cordon"

    def test_remediation_action_fields(self):
        pred = self._make_pred()
        action = remediate(pred, None)
        assert action is not None
        assert action.namespace == "production"
        assert action.deployment == "test-service"
        assert action.timestamp > 0
        assert action.dry_run is False

    def test_cooldown_prevents_action(self):
        pred = self._make_pred()
        mock_rc = MagicMock()
        mock_rc.get.return_value = "1"  # pod is in cooldown
        result = remediate(pred, mock_rc)
        assert result is None

    def test_no_auto_remediate_forces_alert_only(self):
        pred = self._make_pred(risk_level="critical", cause="oom", score=0.95)
        import app
        original = app.AUTO_REMEDIATE
        app.AUTO_REMEDIATE = False
        try:
            action = remediate(pred, None)
            assert action is not None
            assert action.action_type == "alert_only"
        finally:
            app.AUTO_REMEDIATE = original


# ── COOLDOWN TESTS ───────────────────────────────────────────────
class TestCooldown:
    def test_no_redis_not_cooling_down(self):
        assert is_cooling_down("pod1", None) is False

    def test_cooling_down_with_redis(self):
        mock_rc = MagicMock()
        mock_rc.get.return_value = "1"
        assert is_cooling_down("pod1", mock_rc) is True

    def test_not_cooling_down_with_redis(self):
        mock_rc = MagicMock()
        mock_rc.get.return_value = None
        assert is_cooling_down("pod1", mock_rc) is False

    def test_set_cooldown_calls_redis(self):
        mock_rc = MagicMock()
        set_cooldown("pod1", mock_rc)
        mock_rc.set.assert_called_once()

    def test_set_cooldown_no_redis(self):
        # Should not raise
        set_cooldown("pod1", None)


# ── STORE ACTION TESTS ───────────────────────────────────────────
class TestStoreAction:
    def test_store_action_calls_redis(self):
        mock_rc = MagicMock()
        action = RemediationAction(
            pod="pod1", namespace="ns1", deployment="dep1", node="n1",
            action_type="restart", risk_level="high", risk_score=0.85,
            predicted_cause="cpu", eta_minutes=10, dry_run=False,
            result="success", detail="test", timestamp=time.time(),
        )
        store_action(action, mock_rc)
        mock_rc.lpush.assert_called_once()
        mock_rc.ltrim.assert_called_once()
        assert mock_rc.hincrby.call_count == 2

    def test_store_action_no_redis(self):
        action = RemediationAction(
            pod="pod1", namespace="ns1", deployment="dep1", node="n1",
            action_type="restart", risk_level="high", risk_score=0.85,
            predicted_cause="cpu", eta_minutes=10, dry_run=False,
            result="success", detail="test", timestamp=time.time(),
        )
        # Should not raise
        store_action(action, None)


# ── REMEDIATION ACTION DATACLASS TESTS ───────────────────────────
class TestRemediationAction:
    def test_serialization(self):
        action = RemediationAction(
            pod="pod1", namespace="ns1", deployment="dep1", node="n1",
            action_type="scale_up", risk_level="critical", risk_score=0.95,
            predicted_cause="oom", eta_minutes=5, dry_run=False,
            result="success", detail="Scaled dep1: 2 -> 3 replicas",
            timestamp=1234567890.0,
        )
        d = asdict(action)
        assert d["pod"] == "pod1"
        assert d["risk_score"] == 0.95
        assert d["action_type"] == "scale_up"
        # Should be JSON serializable
        json_str = json.dumps(d)
        assert "pod1" in json_str

    def test_all_fields_present(self):
        action = RemediationAction(
            pod="p", namespace="n", deployment="d", node="nd",
            action_type="restart", risk_level="high", risk_score=0.8,
            predicted_cause="cpu", eta_minutes=None, dry_run=True,
            result="dry_run", detail="DRY-RUN", timestamp=0.0,
        )
        d = asdict(action)
        expected_keys = [
            "pod", "namespace", "deployment", "node", "action_type",
            "risk_level", "risk_score", "predicted_cause", "eta_minutes",
            "dry_run", "result", "timestamp", "detail",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"
