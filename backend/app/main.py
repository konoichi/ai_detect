# Backend v0.1 - main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import detect
from .utils.selftest import run_selftest

app = FastAPI(title="KI Detector Backend v0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router, prefix="/api/v1")

@app.get("/api/v1/selftest")
def selftest():
    return run_selftest()
