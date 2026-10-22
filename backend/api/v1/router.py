from fastapi import APIRouter
from api.v1.endpoints import auth, health, jobs, detect, detect_video, detect_audio

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(jobs.router)
api_router.include_router(detect.router)
api_router.include_router(detect_video.router,  prefix="/detect")
api_router.include_router(detect_audio.router,  prefix="/detect")
