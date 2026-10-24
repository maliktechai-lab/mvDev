# DeepGuard — Design, Root-Cause Analysis & Requirements

**Date:** 2026-10-24  
**Status:** Living document — updated after each phase

---

## 1. Repository Audit Summary

### 1.1 What exists and works

| Component | Status | Notes |
|---|---|---|
| FastAPI backend | ✅ Importable | All modules compile; health, auth, jobs, detect endpoints present |
| PostgreSQL + Alembic | ✅ Migration exists | `d4b7fe266094_initial_schema.py` covers all 4 tables |
| Redis | ✅ Config correct | URL settings consistent across config/celery/compose |
| Celery worker | ✅ Registers tasks | video + audio pipelines imported; routes configured |
| Docker Compose | ✅ Valid config | `docker-compose config` passes; service-specific health checks in place |
| Frontend | ✅ Builds | Vite + React 18; API client, auth flow, detect/history/settings pages |
| Auth endpoints | ✅ Complete | Register, login, refresh (rotation), logout, change-password |
| Model readiness gate | ✅ Present | `core/model_readiness.py` + `require_model_ready()` in detect.py |
| Image pipeline | ✅ Logic correct | 3-branch model, face detection, GradCAM |
| Security: upload magic bytes | ✅ Present | `_validate_image_magic()` in detect.py |
| Security: JWT hashing | ✅ Present | Refresh tokens stored hashed |

### 1.2 Confirmed bugs and gaps

| ID | Severity | Component | Issue |
|---|---|---|---|
| B-01 | **Critical** | `models/registry.py` | Silent random-weight fallback in production — `get_image_model()` / `get_video_model()` / `get_audio_model()` call `build_*_model()` on checkpoint failure instead of raising |
| B-02 | **Critical** | `pipelines/video_pipeline.py` | Celery task uses no DB session pattern (no `AsyncSession` in sync task context); likely broken |
| B-03 | **High** | `api/v1/endpoints/detect.py` | DB session not committed after writing job+result — relies on `get_db()` autocommit which only triggers on clean exit; exception path never commits job status update |
| B-04 | **High** | No tests | `pytest --collect-only` = 0 tests collected; no `tests/` directory |
| B-05 | **Medium** | `models/registry.py` | `warmup_models()` swallows all model errors silently — hides B-01 at startup |
| B-06 | **Medium** | `core/config.py` | `IMAGE_MODEL_PATH` defaults to `../ml/checkpoints/deepguard_image.pth` (relative to cwd) — broken when backend/ is cwd inside Docker |
| B-07 | **Medium** | Docker | Celery worker health check: `celery -A core.celery_app inspect ping` — must be run from `/app` with correct PYTHONPATH; not verified inside the actual container image |
| B-08 | **Medium** | `pipelines/video_pipeline.py` | No model readiness gate before inference |
| B-09 | **Low** | `api/v1/endpoints/detect.py` | Response does not tag results with `random_weights_warning` when running in dev mode without a checkpoint |
| B-10 | **Low** | Frontend | `DashboardPage` / `DetectPage` have no model-unavailable banner or handling of HTTP 503 from detect endpoint |
| B-11 | **Low** | `api/v1/endpoints/detect_video.py` | Video file read entirely into memory before size check — risk for large files |
| B-12 | **Low** | Security | CORS `allow_headers=["*"]` is overly permissive; acceptable for dev but should be tightened for prod |
| B-13 | **Low** | `backend/training/train_image.py` | Importable but checkpoint path hard-codes `ml/checkpoints/` relative to working dir |

---

## 2. Root-Cause Analysis

### B-01 — Silent random-weight fallback (registry.py)

**Cause:** Every `get_*_model()` function has two paths:
1. Checkpoint exists → load → on exception: `build_*_model()` (random weights, silent)
2. Checkpoint missing → `build_*_model()` (random weights, with only a `logger.warning`)

The `require_model_ready()` gate in `model_readiness.py` is correct and enforces production safety,
but it is only called in the image detect endpoint. The registry itself never raises.

**Impact:** Any component that calls `get_image_model()` directly (e.g., old code, tests, future 
endpoints) bypasses the gate and gets silent random-weight inference.

**Fix:** Make `get_*_model()` raise `RuntimeError` in production when checkpoint is missing or corrupt.
In development, allow random weights but return them alongside a clear flag. Also remove the
silent exception swallow on checkpoint load failure.

### B-02 — Video pipeline Celery task + DB session

**Cause:** Celery tasks are synchronous Python functions. The FastAPI `AsyncSession` dependency
cannot be used inside them. The video pipeline must use a separate synchronous SQLAlchemy session
or a new async session created within the task.

**Fix:** The Celery task must create its own DB session using `SessionLocal` (sync) or
run an async event loop with `AsyncSessionLocal`.

### B-03 — Detect endpoint DB commit gap

**Cause:** `get_db()` commits on clean exit and rolls back on exception. But the detect endpoint
catches the inference exception, updates `job.status = FAILED`, and then re-raises. At that point
`get_db()` sees the re-raise and rolls back — the FAILED status update is lost.

**Fix:** Explicitly commit the job status update before re-raising, or use a separate 
`try/finally` pattern to persist the status regardless of outcome.

### B-06 — Relative model path broken in Docker

**Cause:** `IMAGE_MODEL_PATH=../ml/checkpoints/deepguard_image.pth` is relative to the process
working directory. Inside Docker, `WORKDIR` is `/app` (the `backend/` folder). The path resolves
to `/ml/checkpoints/`, which does not exist. The volume mount is `/app/checkpoints` (mapping
`./ml/checkpoints`), so the correct path inside the container is `./checkpoints/deepguard_image.pth`
or `/app/checkpoints/deepguard_image.pth`.

**Fix:** The `.env` file (or `.env.example`) must set model paths to `./checkpoints/deepguard_*.pth`
for Docker, or `ml/checkpoints/deepguard_*.pth` for local dev from the repo root. Document both.

---

## 3. Requirements (by Phase)

### Phase 1 — Runtime Stability

**R1.1** Every Docker service health check must reflect that service's own health only.  
**R1.2** The Celery worker health check must use a command available inside the image.  
**R1.3** The backend health endpoint must check DB and Redis connectivity.  
**R1.4** Model paths must resolve correctly in both Docker (WORKDIR=/app) and local dev.  

### Phase 2 — Model Readiness & Safe Inference

**R2.1** `models/registry.py`: In production, missing or corrupt checkpoint must raise `RuntimeError`.  
**R2.2** `models/registry.py`: In development, missing checkpoint must log a loud warning and return the model with a `random_weights=True` attribute.  
**R2.3** `core/model_readiness.py` already implements correct `require_model_ready()` — registry must be aligned.  
**R2.4** `warmup_models()` must not swallow errors silently; it must log them at ERROR level.  
**R2.5** Tests must cover: (a) missing checkpoint → UNAVAILABLE, (b) corrupt file → CORRUPT, (c) valid checkpoint → READY.  

### Phase 3 — Image Detection Workflow

**R3.1** DB commit must be guaranteed even on inference failure (job status = FAILED persisted).  
**R3.2** Dev-mode responses must include `random_weights_warning: true` when running without a checkpoint.  
**R3.3** All error responses must follow `{"detail": "..."}` schema.  
**R3.4** Accepted file types validated by magic bytes, not just MIME.  

### Phase 4 — Video Detection

**R4.1** Model readiness gate must be applied before accepting video jobs.  
**R4.2** Celery task must manage its own DB session (not share FastAPI's AsyncSession).  
**R4.3** Temp files must be cleaned up on task success and failure.  
**R4.4** Failed jobs must persist their error message to the DB.  

### Phase 5 — Security

**R5.1** Upload path must not allow directory traversal.  
**R5.2** File size must be enforced before reading full content (video endpoint reads all to memory).  
**R5.3** CORS `allow_headers` should be scoped in production.  
**R5.4** Sensitive fields (passwords, tokens) must never appear in logs.  

### Phase 6 — Frontend

**R6.1** Detect page must handle HTTP 503 (model unavailable) with a clear, actionable message.  
**R6.2** Dashboard must show model readiness status from `/api/v1/health/models`.  
**R6.3** All API error `detail` fields must be surfaced to the user.  
**R6.4** Frontend production build must succeed without errors.  

### Phase 7 — ML Pipeline

**R7.1** `train_image.py` must be importable and its checkpoint-save path must be configurable.  
**R7.2** Training config YAML must match the actual model constructor signature.  
**R7.3** Training documentation must cover: dataset download, preprocessing, training, evaluation.  

### Phase 8 — Integration & Documentation

**R8.1** Full pytest suite passes (or failures are documented with root cause).  
**R8.2** Frontend build succeeds.  
**R8.3** README covers: setup, env config, Docker startup, migration, model paths, training, testing.  

---

## 4. Architecture Decisions (retained from original)

- **Async FastAPI** with `asyncpg` — kept as-is; correct for I/O-heavy API
- **Celery for video/audio** — correct; image is synchronous (< 2s), video/audio are async
- **Three-branch model from scratch** — no pretrained backbone; training required for meaningful results
- **GradCAM explainability** — wired into image pipeline; appropriate for spatial CNN branches
- **JWT with refresh rotation** — correct security pattern; token hashes stored in DB
- **`model_readiness.py` separate from `health.py`** — correct architecture; keeps app health and ML health independent

---

## 5. Blocked Items (require external resources)

| Item | Blocker | What's needed |
|---|---|---|
| Meaningful deepfake detection | No trained model checkpoint | GPU machine + FaceForensics++ dataset; see README training section |
| GradCAM on real detections | Depends on trained model | Same as above |
| Audio pipeline validation | No audio checkpoint + no evaluation | Treat as experimental |
| End-to-end detection accuracy | No ground-truth test set | Download FF++ or Celeb-DF, run `python ml/datasets/prepare_ff.py` |

---

## 6. File Change Log

| File | Change | Phase |
|---|---|---|
| `docs/DESIGN.md` | Created | Audit |
| `backend/models/registry.py` | Remove silent fallback in production | 1/2 |
| `backend/pipelines/video_pipeline.py` | Fix DB session, add readiness gate | 4 |
| `backend/api/v1/endpoints/detect.py` | Fix DB commit, add warning tag | 3 |
| `backend/core/config.py` | Fix default model paths | 1 |
| `tests/` | Created test suite | 2 |
| `frontend/src/components/common/ModelStatusBanner.tsx` | Add model status banner | 6 |
| `frontend/src/pages/DetectPage.tsx` | Handle 503, show warning | 6 |
| `README.md` | Full rewrite | 8 |
