# DeepGuard — Deepfake & Synthetic Media Detection Platform

A production-grade platform for detecting deepfakes in images, videos, and audio using
multi-branch deep learning models.

> **Status:** Infrastructure complete and tested. Trained model checkpoints are required
> for meaningful detection results — see [Training Models](#training-models) below.

---

## Architecture

```
User → Nginx → FastAPI (JWT auth) → ML Pipelines → PostgreSQL
                        ↓
                 Redis + Celery  (async video / audio jobs)
```

| Layer | Technology |
|---|---|
| ML | PyTorch 2.3, custom three-branch architectures |
| API | FastAPI, SQLAlchemy (async), Alembic |
| Queue | Celery 5.4 + Redis 7 |
| Database | PostgreSQL 16 |
| Frontend | React 18, TypeScript, Tailwind CSS, Vite |
| Auth | JWT (python-jose, bcrypt, refresh-token rotation) |
| Containers | Docker + Docker Compose |
| Proxy | Nginx |
| Explainability | GradCAM |

---

## Quick Start (Docker)

### Prerequisites

- Docker ≥ 24 and `docker-compose` (standalone) — verified with `docker-compose version`
- **Not** `docker compose` (the plugin) — this repo targets the standalone binary

### 1. Clone and configure

```bash
git clone <repo-url>
cd deepguard
cp .env.example .env
# Edit .env — change SECRET_KEY and JWT_SECRET_KEY to long random strings
```

### 2. Start all services

```bash
docker-compose up --build
```

Services:

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (dev) | http://localhost:8000/api/docs |
| Celery Flower | http://localhost:5555 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

### 3. Run database migrations

```bash
docker-compose exec backend alembic upgrade head
```

### 4. Check service health

```bash
# Application health (DB + Redis)
curl http://localhost:8000/api/v1/health

# Model checkpoint readiness
curl http://localhost:8000/api/v1/health/models

# Minimal liveness ping
curl http://localhost:8000/api/v1/health/ping
```

---

## Local Development (without Docker)

### Backend

```bash
# Python 3.11+ required
cd deepguard
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Point .env at local services
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/deepguard
# REDIS_URL=redis://localhost:6379/0
# IMAGE_MODEL_PATH=./checkpoints/deepguard_image.pth  (see Model Paths below)

cd backend
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
npm run build        # production build check
```

### Celery Worker

```bash
cd backend
celery -A core.celery_app worker --loglevel=info -Q video,audio,default
```

### Run tests

```bash
# From the repo root (pytest.ini configures pythonpath=backend)
source .venv/bin/activate
python -m pytest tests/ -v
```

---

## Model Checkpoints

DeepGuard models are trained entirely from scratch — no pretrained backbones or external weights.
Trained checkpoints are **required** for meaningful detection. Without them the API will:

- In **development** mode (`APP_ENV=development`): serve predictions from random weights and tag
  every response with `"random_weights_warning": true`. The dashboard and detection pages show a
  warning banner. Results are meaningless.
- In **production** mode (`APP_ENV=production`): return HTTP 503 immediately for any detect
  request. The application starts and health checks pass, but inference is gated.

### Model paths

| Environment | Path in container | Where files live on host |
|---|---|---|
| Docker | `/app/checkpoints/deepguard_image.pth` | `./ml/checkpoints/deepguard_image.pth` (volume-mounted) |
| Local dev | `./checkpoints/deepguard_image.pth` | Create a symlink: `ln -s ../ml/checkpoints backend/checkpoints` |

Override via `.env`:
```
IMAGE_MODEL_PATH=/app/checkpoints/deepguard_image.pth   # Docker
IMAGE_MODEL_PATH=./checkpoints/deepguard_image.pth      # local dev
```

### Dummy checkpoint (development testing)

Creates a randomly-initialised (untrained) checkpoint in the correct format, so you can test
the full inference pipeline without training:

```bash
source .venv/bin/activate
python backend/utils/create_dummy_checkpoint.py \
    --model image \
    --output ml/checkpoints/deepguard_image.pth
```

⚠️ This produces **meaningless predictions**. Never present dummy-checkpoint results as real
deepfake detection.

---

## Training Models

Trained model weights are not included in this repository. Training requires:

- A labelled dataset (FaceForensics++ or equivalent)
- A machine with a CUDA GPU (8 GB+ VRAM recommended)

### Step 1 — Download FaceForensics++

FaceForensics++ requires registration: https://github.com/ondyari/FaceForensics

```bash
# After obtaining access credentials:
python ml/datasets/prepare_ff.py \
    --source_dir /path/to/raw_ff++ \
    --output_dir ml/datasets/processed
```

This creates `ml/datasets/processed/real/` and `ml/datasets/processed/fake/` directories.

### Step 2 — Train the image model

```bash
# From repo root, GPU machine
source .venv/bin/activate
python backend/training/train_image.py \
    --config ml/experiments/image_model.yaml \
    --dataset_root ml/datasets/processed \
    --device cuda \
    --epochs 50
```

Training parameters (from `ml/experiments/image_model.yaml`):
- Architecture: 3-branch CNN (Spatial + Frequency + Attention), ~31.5M parameters
- Optimizer: AdamW, lr=3e-4, weight decay=1e-4
- Schedule: linear warmup (5 epochs) → cosine decay
- Early stopping: 10 epochs without val AUC improvement
- Augmentation: horizontal flip, colour jitter, Gaussian blur, random erasing

Checkpoint saved to `ml/checkpoints/deepguard_image.pth` automatically.

### Step 3 — Evaluate

Evaluation metrics are logged during training (AUC-ROC, accuracy, F1, EER, confusion matrix).
A separate evaluation pass is printed at the end of training.

### Step 4 — Load into the server

The checkpoint is volume-mounted into the container at `/app/checkpoints/`. After training:

```bash
# Restart the backend to load the new weights
docker-compose restart backend celery_worker
```

Verify the checkpoint loaded:
```bash
curl http://localhost:8000/api/v1/health/models
# "overall": "partial"  → image ready, video/audio still unavailable
```

### Remote GPU training

If training on a separate machine:

```bash
# On GPU machine — train and copy checkpoint back
scp gpu-machine:/path/to/deepguard/ml/checkpoints/deepguard_image.pth \
    ml/checkpoints/deepguard_image.pth

# Then restart services
docker-compose restart backend celery_worker
```

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login` | Login, returns JWT pair |
| POST | `/api/v1/auth/refresh` | Rotate refresh token |
| POST | `/api/v1/auth/logout` | Revoke refresh token |
| POST | `/api/v1/auth/change-password` | Change password |
| GET | `/api/v1/auth/me` | Current user info |

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health/ping` | Minimal liveness probe |
| GET | `/api/v1/health` | DB + Redis connectivity check |
| GET | `/api/v1/health/models` | ML model checkpoint readiness |

`/health/models` returns per-model status: `ready`, `unavailable`, or `corrupt`. This is
intentionally separate from `/health` — a missing checkpoint does not make the application
unhealthy, it just means certain inference endpoints will return 503.

### Detection

| Method | Endpoint | Media | Mode |
|---|---|---|---|
| POST | `/api/v1/detect/image` | JPEG, PNG, WebP, BMP (≤10 MB) | Synchronous |
| POST | `/api/v1/detect/video` | MP4, WebM, AVI, MOV, MKV (≤500 MB) | Async (Celery) |
| POST | `/api/v1/detect/audio` | MP3, WAV, FLAC, OGG, M4A (≤50 MB) | Async (Celery) |

Image detection returns the result immediately. Video and audio return a `job_id`
(HTTP 202). Poll `GET /api/v1/jobs/{job_id}` until `status == "completed"`.

### Jobs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/jobs` | List jobs (paginated, filterable) |
| GET | `/api/v1/jobs/{id}` | Get job + result |
| DELETE | `/api/v1/jobs/{id}` | Delete job |

---

## Project Structure

```
deepguard/
├── backend/
│   ├── api/v1/
│   │   ├── endpoints/      # auth, health, jobs, detect, detect_video, detect_audio
│   │   └── schemas/        # Pydantic request/response models
│   ├── core/               # config, logging, security, celery, model_readiness
│   ├── db/                 # SQLAlchemy models, async session, Alembic migrations
│   ├── models/             # PyTorch model definitions + registry
│   ├── pipelines/          # Image, video, audio inference pipelines
│   ├── preprocessing/      # Face detection, frame extraction, audio processing
│   ├── training/           # Training scripts, dataset loaders, metrics
│   └── utils/              # GradCAM, create_dummy_checkpoint
├── frontend/
│   └── src/
│       ├── pages/          # Dashboard, Detect, History, Settings, Login, Register
│       ├── components/     # Layout, Upload, Results, Analysis, Common
│       ├── store/          # Zustand auth store
│       ├── lib/            # Axios API client (with token refresh interceptor)
│       └── types/          # TypeScript type definitions
├── ml/
│   ├── checkpoints/        # Trained model weights (.pth files go here)
│   ├── datasets/           # Dataset download/preprocessing scripts
│   └── experiments/        # Training configuration YAMLs
├── tests/                  # pytest suite (29 tests)
├── docker/                 # Dockerfiles (backend, frontend)
├── nginx/                  # Nginx reverse proxy config
├── docs/                   # DESIGN.md — architecture, bug log, requirements
├── pytest.ini
├── docker-compose.yml
└── .env.example
```

---

## Testing

```bash
# All tests (no external services required — mocked)
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_model_readiness.py -v
python -m pytest tests/test_auth.py -v
python -m pytest tests/test_image_pipeline.py -v
```

Test coverage by module:

| File | Tests | What's covered |
|---|---|---|
| `test_model_readiness.py` | 7 | UNAVAILABLE / CORRUPT / READY states, prod/dev gating |
| `test_registry.py` | 4 | Registry dev/prod behaviour, warmup error handling |
| `test_auth.py` | 5 | Password hashing, JWT create/decode, token type checks |
| `test_health.py` | 3 | /ping, /models schema, all-unavailable state |
| `test_image_pipeline.py` | 6 | Pipeline result shape, corrupt input, 503/415/413 HTTP |
| `test_celery_tasks.py` | 3 | Task importability, Celery config |

---

## Environment Variables

See `.env.example` for the full annotated list. Key variables:

| Variable | Description | Example |
|---|---|---|
| `APP_ENV` | `development` or `production` | `development` |
| `SECRET_KEY` | App secret (change in prod) | 64+ random hex chars |
| `DATABASE_URL` | PostgreSQL async URL | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis URL | `redis://redis:6379/0` |
| `JWT_SECRET_KEY` | JWT signing key (change in prod) | 64+ random hex chars |
| `IMAGE_MODEL_PATH` | Path to image checkpoint | `/app/checkpoints/deepguard_image.pth` |
| `DEVICE` | PyTorch device | `cpu` or `cuda` |

---

## Docker Compose Health Checks

Each service has a health check appropriate to what it actually runs:

| Service | Health check | Why |
|---|---|---|
| `backend` | HTTP GET `/api/v1/health/ping` | FastAPI is the HTTP server |
| `celery_worker` | `celery inspect ping` | Worker has no HTTP server |
| `flower` | HTTP GET `localhost:5555/healthcheck` | Flower is its own HTTP server |
| `postgres` | `pg_isready` | Built-in PostgreSQL tool |
| `redis` | `redis-cli ping` | Built-in Redis tool |

---

## Security Notes

- JWT refresh tokens are stored hashed (SHA-256) in PostgreSQL — raw tokens never persisted
- Upload filenames are replaced with `uuid4().hex + safe_extension` — no user-controlled paths written to disk
- File content validated by magic bytes (not just MIME type) for image uploads
- `Content-Length` pre-check on video uploads prevents full memory read before size rejection
- CORS `allow_origins` driven by `ALLOWED_ORIGINS` env var — restrict this in production
- **No brute-force protection on `/auth/login`** — add rate limiting (e.g. `slowapi`) before production deployment
- Run behind Nginx with `client_max_body_size` set to enforce upload limits at the proxy layer

---

## Limitations

| Item | Status |
|---|---|
| Image detection | Pipeline complete. **Requires trained checkpoint for real results.** |
| Video detection | Pipeline complete (Celery async, frame extraction, LSTM). **Requires trained checkpoint.** |
| Audio detection | Pipeline present. Marked experimental — no trained checkpoint, no evaluation. |
| Model accuracy | Not measured — no trained model has been evaluated on a held-out test set. |
| GradCAM | Wired into image pipeline. Requires trained model to produce meaningful heatmaps. |
| GPU inference | Supported via `DEVICE=cuda`. CPU is the default for development. |
| Brute-force protection | Not implemented — add `slowapi` or Nginx `limit_req` before production. |

---

## Troubleshooting

### Backend won't start — "Model X is not available in production"

Set `APP_ENV=development` in `.env`, or provide trained checkpoints at the configured paths.

### Celery worker health check failing

The `celery inspect ping` command requires the worker to be connected to Redis. Check that
`CELERY_BROKER_URL` in `.env` matches the Redis service name in `docker-compose.yml`.

```bash
docker-compose logs celery_worker | tail -20
```

### Database migrations not applied

```bash
docker-compose exec backend alembic upgrade head
```

### Frontend shows "Model not available" banner

This is expected in a fresh environment with no trained checkpoints. See
[Model Checkpoints](#model-checkpoints) above.

### Docker Compose command not found

This repo uses the standalone `docker-compose` binary, not the `docker compose` plugin.
Install it: https://docs.docker.com/compose/install/standalone/
