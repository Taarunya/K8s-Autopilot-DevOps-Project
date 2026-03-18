# K8s Autopilot

**AI-powered Kubernetes failure prediction and auto-remediation engine.**

K8s Autopilot monitors your cluster pods in real time, predicts failures before they happen using machine learning, and automatically takes corrective action — scaling replicas, restarting deployments, bumping memory limits, or cordoning nodes.

---

## Architecture

```
┌──────────────┐     metrics     ┌──────────────┐
│   Exporter   │ ──────────────► │  Prometheus  │
│   :8000      │                 │   :9090      │
└──────────────┘                 └──────┬───────┘
                                        │ query_range
                                        ▼
                                 ┌──────────────┐     ┌─────────┐
                                 │  Predictor   │◄───►│  Redis  │
                                 │   :8001      │     │  :6379  │
                                 └──────┬───────┘     └────┬────┘
                                        │ predictions      │
                                        ▼                  │
                                 ┌──────────────┐          │
                                 │   Operator   │◄─────────┘
                                 │   :8002      │
                                 └──────┬───┬───┘
                                        │   │
                             K8s API ◄──┘   └──► Slack
                                        
                                 ┌──────────────┐
                                 │  Dashboard   │ proxies Predictor + Operator
                                 │   :3000      │
                                 └──────────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **Exporter** | 8000 | Simulates Kubernetes pod metrics (CPU, memory, network) with realistic daily patterns and anomaly injection (OOM leaks, CPU spikes, network storms) |
| **Predictor** | 8001 | ML pipeline using Gradient Boosting + Isolation Forest to score pod risk. Extracts 12 features from Prometheus time-series windows |
| **Operator** | 8002 | Decision engine that maps (risk_level × predicted_cause) to remediation actions. Supports real K8s API, mock mode, and dry-run |
| **Dashboard** | 3000 | Real-time monitoring UI with risk overview, pod predictions table, actions log, node topology, and model analytics |
| **Prometheus** | 9090 | Time-series database scraping metrics from the exporter and predictor |
| **Redis** | 6379 | Shared state for predictions, action history, cooldowns, and model metrics |

### ML Pipeline

- **Feature Engineering:** Memory utilization (mean, P99, slope, acceleration), CPU utilization & throttle, network error rate & spikes, restart rate, OOM history, anomaly score
- **Models:** GradientBoostingClassifier (risk classification) + IsolationForest (unsupervised anomaly detection)
- **Training:** Synthetic data generation with realistic normal/failure distributions. Auto-retrains on demand

### Remediation Decision Matrix

| Risk Level | Cause: OOM | Cause: CPU | Cause: Network | Cause: Unknown |
|-----------|------------|------------|----------------|----------------|
| **Critical** | Scale up replicas | Cordon node | Rolling restart | Rolling restart |
| **High** | Bump memory limit | Rolling restart | Rolling restart | Rolling restart |
| **Medium** | Bump memory limit | Bump memory limit | Bump memory limit | Bump memory limit |
| **Low** | Alert only | Alert only | Alert only | Alert only |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) A Kubernetes cluster with `kubectl` configured for live remediation
- (Optional) A Slack webhook URL for alerts

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your Slack webhook URL and API key
```

### 2. Launch

```bash
docker-compose up --build -d
```

### 3. Access

| URL | Service |
|-----|---------|
| [http://localhost:3000](http://localhost:3000) | Dashboard UI |
| [http://localhost:9090](http://localhost:9090) | Prometheus |
| [http://localhost:8001/docs](http://localhost:8001/docs) | Predictor API docs |
| [http://localhost:8002/docs](http://localhost:8002/docs) | Operator API docs |

---

## Configuration

All configuration is via environment variables in `docker-compose.yml` (or `.env`):

### Exporter
| Variable | Default | Description |
|----------|---------|-------------|
| `MOCK_MODE` | `true` | Enable simulated metrics |
| `NUM_PODS` | `30` | Number of simulated pods |
| `INJECT_ANOMALIES` | `true` | Inject OOM/CPU/network anomalies |

### Predictor
| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_URL` | `http://prometheus:9090` | Prometheus endpoint |
| `REDIS_URL` | `redis://redis:6379` | Redis endpoint |
| `PREDICTION_INTERVAL` | `30` | Seconds between prediction cycles |
| `RISK_THRESHOLD_HIGH` | `0.75` | Risk score threshold for "high" |
| `RISK_THRESHOLD_MEDIUM` | `0.50` | Risk score threshold for "medium" |

### Operator
| Variable | Default | Description |
|----------|---------|-------------|
| `MOCK_K8S` | `false` | Use mock K8s actions (no real API calls) |
| `AUTO_REMEDIATE` | `true` | Enable auto-remediation (false = alert only) |
| `DRY_RUN` | `false` | Log actions without executing |
| `COOLDOWN_SECS` | `300` | Seconds before re-remediating same pod |
| `SLACK_WEBHOOK_URL` | — | Slack incoming webhook for alerts |
| `API_KEY` | — | API key for authenticated endpoints |

---

## API Endpoints

### Predictor (`:8001`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health + model status |
| GET | `/predictions` | All pod predictions (filterable by `risk_level`, `namespace`) |
| GET | `/predictions/summary` | Aggregated risk and cause counts |
| GET | `/history` | High-risk prediction history |
| GET | `/model/metrics` | Model performance metrics |
| POST | `/retrain` | Trigger model retraining (API key required) |

### Operator (`:8002`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Operator health + stats |
| GET | `/actions` | Remediation action log |
| GET | `/actions/summary` | Action type & result aggregates |
| DELETE | `/actions/clear` | Clear action history (API key required) |

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=predictor --cov=operator
```

---

## Project Structure

```
k8s-autopilot/
├── docker-compose.yml      # Service orchestration
├── .env.example            # Configuration template
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml          # CI/CD pipeline
├── exporter/               # Metrics simulator
│   ├── Dockerfile
│   ├── exporter.py
│   └── requirements.txt
├── predictor/              # ML prediction service
│   ├── Dockerfile
│   ├── main.py
│   ├── predictor.py
│   ├── fetcher.py
│   └── requirements.txt
├── operator/               # Remediation engine
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── dashboard/              # Monitoring UI
│   ├── Dockerfile
│   ├── server.py
│   ├── index.html
│   └── requirements.txt
├── k8s/                    # Kubernetes manifests
│   ├── namespace.yaml
│   ├── redis.yaml
│   ├── prometheus.yaml
│   ├── exporter.yaml
│   ├── predictor.yaml
│   ├── operator.yaml
│   └── dashboard.yaml
├── docker/
│   └── prometheus.yml      # Prometheus config
└── tests/
    ├── test_predictor.py
    └── test_operator.py
```

---

## License

This project is licensed under the MIT License.

## Author

Taarunya Aggarwal

GitHub: @Taarunya
Email: taru.agg05@gmail.com
