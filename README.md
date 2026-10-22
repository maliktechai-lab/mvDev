# DeepGuard — Deepfake & Synthetic Media Detection Platform

A production-grade platform to detect deepfakes in images, videos, and audio using multi-branch deep learning models.

---

## Architecture Overview

```
User → Nginx → FastAPI (JWT auth) → ML Pipelines → PostgreSQL
                          ↓
                    Redis + Celery (async video/audio jobs)
```

---

## Models

| Model | Architecture | Media | Branches |
|---|---|---|---|
| DeepGuard-Image | EfficientNet-B4 | Image | Spatial + Frequency + Attention |
| DeepGuard-Video | EfficientNet + LSTM | Video | Spatial + Frequency + Temporal + Biological (rPPG) |
| DeepGuard-Audio | RawNet2 | Audio | Raw Waveform + Spectral + Prosody |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 20+ (for frontend development)

### 1. Clone & configure
```bash
git clone <repo>
cd deepguard
cp .env.example .env
# Edit .env with your secrets
```

### 2. Start all services
```bash
docker-compose up --build
```

Services started:
- **Frontend** → http://localhost:3000
- **Backend API** → http://localhost:8000
- **API Docs** → http://localhost:8000/api/docs (dev mode)
- **Celery Flower** → http://localhost:5555
- **PostgreSQL** → localhost:5432
- **Redis** → localhost:6379

### 3. Run database migrations
```bash
docker-compose exec backend alembic upgrade head
```

---

## Local Development

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

### Celery Worker
```bash
cd backend
celery -A core.celery_app worker --loglevel=info
```

---

## Project Structure

```
deepguard/
├── backend/
│   ├── api/v1/
│   │   ├── endpoints/      # auth, health, jobs, detect (Phase 2+)
│   │   └── schemas/        # Pydantic request/response models
│   ├── core/               # config, logging, security, celery
│   ├── db/                 # SQLAlchemy models, migrations
│   ├── models/             # PyTorch model definitions
│   ├── pipelines/          # Image, video, audio inference pipelines
│   ├── preprocessing/      # Face detection, frame extraction, audio processing
│   ├── training/           # Training scripts and data loaders
│   └── utils/              # GradCAM, metrics, file utilities
├── frontend/
│   └── src/
│       ├── pages/          # Dashboard, Detect, History, Settings
│       ├── components/     # Layout, Upload, Results, Common
│       ├── store/          # Zustand state management
│       ├── lib/            # Axios API client
│       └── types/          # TypeScript type definitions
├── ml/
│   ├── checkpoints/        # Trained model weights (.pth files)
│   ├── datasets/           # Dataset download scripts
│   └── experiments/        # Training configuration YAMLs
├── docker/                 # Dockerfiles
├── nginx/                  # Nginx configs
└── docker-compose.yml
```

---

## Build Phases

- **Phase 1** ✅ — Foundation: Docker, FastAPI, Auth, Database, React UI
- **Phase 2** 🔄 — Image detection: EfficientNet-B4 model + 3 branches + GradCAM
- **Phase 3** ⏳ — Video detection: EfficientNet + LSTM + rPPG + Celery
- **Phase 4** ⏳ — Audio detection: RawNet2 + 3 branches

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/auth/register | Create account |
| POST | /api/v1/auth/login | Get tokens |
| POST | /api/v1/auth/refresh | Refresh access token |
| POST | /api/v1/auth/logout | Revoke refresh token |
| GET | /api/v1/auth/me | Current user info |
| GET | /api/v1/health | Health check |
| GET | /api/v1/jobs | List user jobs |
| GET | /api/v1/jobs/{id} | Get job + result |
| DELETE | /api/v1/jobs/{id} | Delete job |
| POST | /api/v1/detect/image | **(Phase 2)** Detect fake image |
| POST | /api/v1/detect/video | **(Phase 3)** Submit video for detection |
| POST | /api/v1/detect/audio | **(Phase 4)** Submit audio for detection |

---

## Environment Variables

See `.env.example` for all required variables with descriptions.

---

## Training Models

When GPU is available:
```bash
# Download FaceForensics++ dataset
python ml/datasets/download_ff.py

# Train image model
python backend/training/train_image.py --config ml/experiments/image_model.yaml

# Checkpoint saved to ml/checkpoints/deepguard_image.pth
# Restart backend to load new weights
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | PyTorch 2.3, timm, facenet-pytorch |
| API | FastAPI, SQLAlchemy, Alembic |
| Queue | Celery + Redis |
| Database | PostgreSQL 16 |
| Frontend | React 18, TypeScript, Tailwind CSS, Vite |
| Auth | JWT (python-jose, bcrypt) |
| Containers | Docker + Docker Compose |
| Proxy | Nginx |
| Explainability | GradCAM (grad-cam) |
