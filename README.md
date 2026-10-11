# Efficient ML Deployment (Docker + FastAPI + MLOps)

## 📌 Project Overview
Deploy machine learning models as containerized, low-latency microservices with rapid scaling, observability, and serverless patterns.

---

## 🏗️ Repository Architecture
```text
ml-cancer-deployment/
├── .github/workflows/        # CI/CD and benchmark automated pipelines
├── app/                      # FastAPI application code
│   ├── api/v1/               # Versioned endpoints (/health, /predict, /predict/batch)
│   ├── core/                 # Inference engine, Pydantic schemas, telemetry
│   ├── config.py             # Environment configuration (Pydantic Settings)
│   ├── lifespan.py           # Startup/shutdown lifecycle (model pre-warming)
│   └── main.py               # Application entrypoint
├── models/                   # Model storage directory (.gitkeep)
├── scripts/                  # Training, benchmarking, and artifact download scripts
├── tests/                    # Integration, unit, and latency benchmark tests
├── openfaas/                 # Serverless deployment templates (of-watchdog)
├── Dockerfile                # Multi-stage production container build
├── docker-compose.yml        # Local orchestration (API + Prometheus)
├── requirements.txt          # Production dependencies
└── requirements-dev.txt      # Testing and development dependencies
```

---

## 🚀 Quickstart for Team Members

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Train and Export Model Artifacts
```bash
python scripts/train.py
```

### 3. Run Application Locally
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Run Test Suite
```bash
pytest tests/ -v
```

### 5. Build and Run with Docker
```bash
docker compose up --build
```
