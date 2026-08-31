"""
NAVDRIFT-0 FastAPI Backend.

Endpoints:
  POST /ingest      — receive one IMU+odometry timestep, return DR pose + uncertainty
  POST /reacquire   — receive a GNSS fix, run SNAP correction, return corrected trajectory
  GET  /status      — health check and current state
  GET  /trajectory  — full trajectory history (for visualization)
  POST /gnss_lost   — notify system of GNSS signal loss
  POST /reset       — reset the runtime to a new initial GNSS fix

Security:
  - API key authentication via X-API-Key header (set via environment variable)
  - CORS restricted to configured origins
  - Request body size limited to 64 KB
  - All inputs validated with Pydantic
  - Errors return safe messages — no stack traces in production
  - Rate limiting via slowapi

Run:
    NAVDRIFT_API_KEY=your_secret_key \
    ONNX_PATH=./checkpoints/onnx/drift_former.onnx \
    NORM_STATS_PATH=./checkpoints/drift_former/norm_stats.npz \
    uvicorn navdrift0.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, validator

from inference.runtime import NavDriftRuntime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

NAVDRIFT_API_KEY   = os.environ.get("NAVDRIFT_API_KEY", "")
ONNX_PATH          = os.environ.get("ONNX_PATH", "./checkpoints/onnx/drift_former.onnx")
NORM_STATS_PATH    = os.environ.get("NORM_STATS_PATH", "./checkpoints/drift_former/norm_stats.npz")
ALLOWED_ORIGINS    = os.environ.get("ALLOWED_ORIGINS", "http://localhost:7860").split(",")
WINDOW             = int(os.environ.get("WINDOW", "200"))
IMU_HZ             = float(os.environ.get("IMU_HZ", "100.0"))
DEMO_MODE          = os.environ.get("DEMO_MODE", "false").lower() == "true"

if not NAVDRIFT_API_KEY:
    if DEMO_MODE:
        # In demo mode, generate a random key that gets printed at startup.
        # Never runs without auth — the key is still required.
        NAVDRIFT_API_KEY = secrets.token_hex(16)
        logger.warning("DEMO MODE: generated ephemeral API key: %s", NAVDRIFT_API_KEY)
    else:
        raise RuntimeError("NAVDRIFT_API_KEY environment variable must be set")

# ---------------------------------------------------------------------------
# Global runtime (lazy-loaded so tests can import without the ONNX file)
# ---------------------------------------------------------------------------

_runtime: Optional[NavDriftRuntime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime
    logger.info("Loading NAVDRIFT-0 runtime...")
    try:
        _runtime = NavDriftRuntime(
            onnx_path        = ONNX_PATH,
            norm_stats_path  = NORM_STATS_PATH,
            window           = WINDOW,
            imu_hz           = IMU_HZ,
        )
        logger.info("Runtime loaded successfully")
    except Exception as exc:
        logger.error("Failed to load runtime: %s", exc)
        # Allow app to start in degraded mode (useful for demo/testing)
        _runtime = None
    yield
    logger.info("NAVDRIFT-0 shutdown")


def get_runtime() -> NavDriftRuntime:
    if _runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check ONNX_PATH and NORM_STATS_PATH.",
        )
    return _runtime


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NAVDRIFT-0",
    description="AI-ML Dead Reckoning API for seamless vehicle navigation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,       # disable redoc in production
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware: CORS, trusted hosts, body size limit
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,      # no cookie-based auth here
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Limit request body to 64 KB
MAX_BODY_SIZE = 64 * 1024  # 64 KB


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large (max 64 KB)"},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: Optional[str] = Depends(api_key_header)) -> None:
    """Verify the X-API-Key header using a constant-time comparison."""
    if not key or not secrets.compare_digest(key.encode(), NAVDRIFT_API_KEY.encode()):
        logger.warning("Unauthorized access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ---------------------------------------------------------------------------
# Pydantic schemas (strict validation — reject extra fields)
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """One IMU + odometry timestep."""
    accel_x:     float = Field(..., ge=-160.0, le=160.0,  description="m/s^2")
    accel_y:     float = Field(..., ge=-160.0, le=160.0,  description="m/s^2")
    accel_z:     float = Field(..., ge=-160.0, le=160.0,  description="m/s^2")
    gyro_x:      float = Field(..., ge=-35.0,  le=35.0,   description="rad/s")
    gyro_y:      float = Field(..., ge=-35.0,  le=35.0,   description="rad/s")
    gyro_z:      float = Field(..., ge=-35.0,  le=35.0,   description="rad/s")
    speed:       float = Field(..., ge=0.0,    le=100.0,  description="m/s")
    steer_angle: float = Field(..., ge=-1.0,   le=1.0,    description="radians")
    timestamp:   Optional[float] = Field(None, ge=0.0)

    class Config:
        extra = "forbid"


class IngestResponse(BaseModel):
    pose_x:            float
    pose_y:            float
    heading_rad:       float
    uncertainty_major: float
    uncertainty_minor: float
    gnss_active:       bool
    step:              int
    latency_ms:        float


class GNSSReacquireRequest(BaseModel):
    latitude:    float = Field(..., ge=-90.0,  le=90.0)
    longitude:   float = Field(..., ge=-180.0, le=180.0)
    heading_deg: float = Field(..., ge=0.0,    le=360.0)

    class Config:
        extra = "forbid"


class TrajectoryPoint(BaseModel):
    x:     float
    y:     float
    theta: float


class ReacquireResponse(BaseModel):
    corrected_trajectory: List[TrajectoryPoint]
    endpoint_error_m:     float
    runtime_ms:           float
    n_steps_corrected:    int


class InitRequest(BaseModel):
    latitude:     float = Field(..., ge=-90.0,  le=90.0)
    longitude:    float = Field(..., ge=-180.0, le=180.0)
    heading_deg:  float = Field(..., ge=0.0,    le=360.0)
    speed_m_s:    float = Field(0.0, ge=0.0, le=100.0)

    class Config:
        extra = "forbid"


class StatusResponse(BaseModel):
    status:       str
    gnss_active:  bool
    step_count:   int
    current_x:    float
    current_y:    float
    model_loaded: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
async def status_check(
    _: None = Depends(verify_api_key),
    runtime: NavDriftRuntime = Depends(get_runtime),
) -> StatusResponse:
    """Health check and current runtime state."""
    pose = runtime.current_pose
    return StatusResponse(
        status       = "ok",
        gnss_active  = runtime.gnss_active,
        step_count   = runtime.step_count,
        current_x    = float(pose[0]),
        current_y    = float(pose[1]),
        model_loaded = True,
    )


@app.post("/init", status_code=200)
async def initialize(
    body:    InitRequest,
    _:       None = Depends(verify_api_key),
    runtime: NavDriftRuntime = Depends(get_runtime),
) -> dict:
    """Set the initial GNSS fix to establish a local coordinate frame."""
    runtime.set_initial_gnss_fix(
        lat         = body.latitude,
        lon         = body.longitude,
        heading_deg = body.heading_deg,
        speed_m_s   = body.speed_m_s,
    )
    return {"message": "Initialized", "ref_lat": body.latitude, "ref_lon": body.longitude}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    body:    IngestRequest,
    _:       None = Depends(verify_api_key),
    runtime: NavDriftRuntime = Depends(get_runtime),
) -> IngestResponse:
    """
    Receive one IMU + odometry timestep and return the dead-reckoning pose estimate.

    Call this at your IMU sampling rate (typically 100 Hz).
    """
    accel = np.array([body.accel_x, body.accel_y, body.accel_z], dtype=np.float32)
    gyro  = np.array([body.gyro_x,  body.gyro_y,  body.gyro_z],  dtype=np.float32)

    result = runtime.ingest(
        accel_xyz   = accel,
        gyro_xyz    = gyro,
        speed       = body.speed,
        steer_angle = body.steer_angle,
        timestamp   = body.timestamp,
    )
    return IngestResponse(**result)


@app.post("/gnss_lost", status_code=200)
async def gnss_lost(
    _:       None = Depends(verify_api_key),
    runtime: NavDriftRuntime = Depends(get_runtime),
) -> dict:
    """Notify the system that the GNSS signal has been lost."""
    runtime.notify_gnss_lost()
    return {"message": "GNSS loss registered", "step": runtime.step_count}


@app.post("/reacquire", response_model=ReacquireResponse)
async def reacquire(
    body:    GNSSReacquireRequest,
    _:       None = Depends(verify_api_key),
    runtime: NavDriftRuntime = Depends(get_runtime),
) -> ReacquireResponse:
    """
    Process a GNSS reacquisition fix.

    Runs SNAP-Corrector to smoothly adjust the dead-reckoning trajectory
    so the endpoint matches the new GNSS fix, with no discontinuity jump.
    """
    result = runtime.reacquire(
        lat         = body.latitude,
        lon         = body.longitude,
        heading_deg = body.heading_deg,
    )
    corrected = [
        TrajectoryPoint(x=pt[0], y=pt[1], theta=pt[2])
        for pt in result["corrected_trajectory"]
    ]
    return ReacquireResponse(
        corrected_trajectory = corrected,
        endpoint_error_m     = result["endpoint_error_m"],
        runtime_ms           = result["runtime_ms"],
        n_steps_corrected    = len(corrected),
    )


@app.get("/trajectory")
async def get_trajectory(
    _:       None = Depends(verify_api_key),
    runtime: NavDriftRuntime = Depends(get_runtime),
) -> dict:
    """Return the full trajectory history for visualization."""
    traj = runtime.get_full_trajectory()
    return {
        "trajectory": [
            {"x": float(p[0]), "y": float(p[1]), "theta": float(p[2])}
            for p in traj
        ],
        "n_points": len(traj),
    }


@app.post("/reset", status_code=200)
async def reset(
    body:    InitRequest,
    _:       None = Depends(verify_api_key),
    runtime: NavDriftRuntime = Depends(get_runtime),
) -> dict:
    """Reset the runtime to a new initial GNSS fix, clearing all history."""
    runtime.pose_history.clear()
    runtime.raw_dr_history.clear()
    runtime.step_count = 0
    runtime.set_initial_gnss_fix(
        lat         = body.latitude,
        lon         = body.longitude,
        heading_deg = body.heading_deg,
        speed_m_s   = body.speed_m_s,
    )
    return {"message": "Runtime reset", "ref_lat": body.latitude, "ref_lon": body.longitude}


# ---------------------------------------------------------------------------
# Production error handler — no stack traces to clients
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url, exc,
                 exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check server logs."},
    )
