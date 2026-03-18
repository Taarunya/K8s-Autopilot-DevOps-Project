# K8s Autopilot — Fixes Applied

## Summary of all bugs fixed

### 1. ✅ Port 3000 conflict → Dashboard now on port 3001
**File:** `docker-compose.yml`  
**Problem:** Port 3000 was already in use on your machine, causing the dashboard container to fail with:
```
Bind for 0.0.0.0:3000 failed: port is already allocated
```
**Fix:** Dashboard now maps `3001:3000` (host:container).  
**Access dashboard at:** http://localhost:3001

---

### 2. ✅ Kubernetes connection errors fixed → MOCK_K8S=true
**File:** `docker-compose.yml`  
**Problem:** `MOCK_K8S=false` was set, so the operator was trying to connect to a real Kubernetes cluster at `kubernetes.docker.internal:6443`, which doesn't exist in this setup. This caused hundreds of `Connection refused` retry errors every cycle.  
**Fix:** Set `MOCK_K8S=true` — the operator now uses mock actions (logs what it *would* do) instead of failing to connect.

---

### 3. ✅ kube config volume mount removed
**File:** `docker-compose.yml`  
**Problem:** The operator service had `${USERPROFILE}/.kube/config:/root/.kube/config:ro` — this uses a Windows-only `USERPROFILE` variable and fails on Linux/Mac. It also crashes if `.kube/config` doesn't exist at all.  
**Fix:** Removed the volume mount entirely since `MOCK_K8S=true` makes it unnecessary.

---

### 4. ✅ Slack webhook placeholder fixed
**File:** `.env`  
**Problem:** `.env` had the literal placeholder string `https://hooks.slack.com/services/YOUR/WEBHOOK/URL` which caused real HTTP 404 errors on every alert (8+ warnings per cycle).  
**Fix:** Set `SLACK_WEBHOOK_URL=` (empty). The operator silently skips Slack when empty instead of spamming 404 errors.

**To add your Slack webhook:** Open `.env` and set:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/REAL/WEBHOOK
```

---

## Service URLs (after running `docker compose up`)
| Service    | URL                        |
|------------|----------------------------|
| Dashboard  | http://localhost:3001      |
| Prometheus | http://localhost:9090      |
| Predictor  | http://localhost:8001      |
| Operator   | http://localhost:8002      |
| Exporter   | http://localhost:8000      |

## Quick Start
```bash
# 1. Edit .env and add your Slack webhook (optional)
# 2. Start everything
docker compose up --build

# Or run in background
docker compose up --build -d
```
