# pbs_bonsai

Monorepo skeleton for the BonsAI project.

```
pbs_bonsai/
├─ frontend/              # Next.js app (handled by the frontend team)
├─ backend/               # FastAPI + MLflow-enabled chat orchestrator
│  ├─ app/
│  │  ├─ api/             # FastAPI endpoints
│  │  ├─ orchestrator/    # Chat orchestrator + scorers
│  │  ├─ data/            # CSV data stores (species, marketplace)
│  │  └─ prompts/         # LLM system prompts
│  ├─ requirements.txt
│  └─ Dockerfile
└─ docker-compose.yml
```

## Quick start (dev)

1. **Install dependencies (optional local run)**
   ```bash
   cd backend
   pip install -r requirements.txt
   export MLFLOW_TRACKING_URI=http://localhost:5001
   ```

2. **Run with Docker Compose (recommended)**
   ```bash
   docker compose up --build
   ```
   - Backend: http://localhost:8080
   - MLflow UI: http://localhost:5001

3. **Test the API**
   ```bash
   curl -X POST http://localhost:8080/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"I am a beginner with 120 EUR, prefer indoor, medium light. I like Ficus.","top_k":5}'
   ```

## MLflow integration

- Experiment name: `bonsai-chat-orchestrator`
- Parameters: `mode`, `k`, `prompt_version` (guided only), etc.
- Metrics: `latency_ms`, `results_count`
- Artifacts: `buyer_profile.json`, `species_candidates.json`, `recommendations.json`

You can change `MLFLOW_TRACKING_URI` via docker-compose or environment variables.
