"""
api/app.py — NAVDRIFT-0 FastAPI backend
Includes HTTP endpoints + WebSocket /ws/stream endpoint.

Environment variables:
  API_KEY      — required; shared secret for X-API-Key / ?api_key auth
  DEMO_MODE    — set to "true" to skip loading the ONNX model (simulated poses)
  CORS_ORIGINS — comma-separated allowed origins (default: *)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import random
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("navdrift0")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY: str = os.environ.get("API_KEY", "")
DEMO_MODE: bool = os.environ.get("DEMO_MODE", "false").strip().lower() == "true"
CORS_ORIGINS: List[str] = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "*").split(",")
    if o.strip()
]
MAX_BODY_BYTES: int = 64 * 1024  # 64 KB
WS_STREAM_HZ: float = 10.0       # background inference rate

if not API_KEY and not DEMO_MODE:
    logger.warning(
        "API_KEY is not set. All requests will be rejected unless DEMO_MODE=true."
    )

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ---------------------------------------------------------------------------
# NavDrift runtime (real or demo)
# ---------------------------------------------------------------------------

class NavDriftRuntime:
    """
    Wraps the ONNX inference session.  In DEMO_MODE the real model is never
    loaded; instead every call returns a deterministic simulated pose.
    """

    def __init__(self) -> None:
        self._session = None
        self._step: int = 0
        self._pose_x: float = 0.0
        self._pose_y: float = 0.0
        self._heading_rad: float = 0.0
        self._gnss_lost: bool = False

    def load(self, model_path: str = "drift_former_int8.onnx") -> None:
        if DEMO_MODE:
            logger.info("DEMO_MODE enabled — skipping ONNX model load.")
            return
        try:
            import onnxruntime as ort  # type: ignore

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNX model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load ONNX model: %s", exc)
            raise RuntimeError("Model load failed") from exc

    # ------------------------------------------------------------------
    # Core inference — returns (delta_x, delta_y, delta_heading)
    # ------------------------------------------------------------------
    def infer(self, sensor: Dict[str, float]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        if DEMO_MODE or self._session is None:
            delta = self._demo_infer(sensor)
        else:
            delta = self._onnx_infer(sensor)

        self._pose_x += delta["dx"]
        self._pose_y += delta["dy"]
        self._heading_rad += delta["dh"]
        self._step += 1
        latency_ms = (time.perf_counter() - t0) * 1_000

        return {
            "pose_x": self._pose_x,
            "pose_y": self._pose_y,
            "heading_rad": self._heading_rad,
            "uncertainty_major": delta.get("unc_major", 0.5),
            "uncertainty_minor": delta.get("unc_minor", 0.3),
            "latency_ms": round(latency_ms, 3),
            "step": self._step,
        }

    def _demo_infer(self, sensor: Dict[str, float]) -> Dict[str, Any]:
        """Simulated pose delta — no real model required."""
        t = time.monotonic()
        dx = 0.05 * math.cos(t * 0.3) + random.gauss(0, 0.002)
        dy = 0.05 * math.sin(t * 0.3) + random.gauss(0, 0.002)
        dh = 0.001 * math.sin(t * 0.1) + random.gauss(0, 0.0001)
        return {"dx": dx, "dy": dy, "dh": dh, "unc_major": 0.5, "unc_minor": 0.3}

    def _onnx_infer(self, sensor: Dict[str, float]) -> Dict[str, Any]:
        """Run real DRIFTFormer inference."""
        keys = [
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z",
            "baro_hpa",
        ]
        imu = np.array(
            [sensor.get(k, 0.0) for k in keys], dtype=np.float32
        ).reshape(1, 1, -1)

        outputs = self._session.run(None, {"imu_seq": imu})
        dx, dy, dh = float(outputs[0][0]), float(outputs[1][0]), float(outputs[2][0])
        unc_major = float(outputs[3][0]) if len(outputs) > 3 else 0.5
        unc_minor = float(outputs[4][0]) if len(outputs) > 4 else 0.3
        return {"dx": dx, "dy": dy, "dh": dh, "unc_major": unc_major, "unc_minor": unc_minor}

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def get_trajectory(self) -> List[Dict[str, float]]:
        # In production this would be stored in a ring buffer.
        return [{"x": self._pose_x, "y": self._pose_y, "heading_rad": self._heading_rad}]

    def mark_gnss_lost(self) -> None:
        self._gnss_lost = True
        logger.info("GNSS marked lost at step %d", self._step)

    def reacquire(self, lat: float, lon: float, heading: float) -> None:
        self._gnss_lost = False
        self._heading_rad = heading
        logger.info("GNSS reacquired at lat=%.6f lon=%.6f heading=%.4f", lat, lon, heading)

    def reset(self) -> None:
        self._step = 0
        self._pose_x = 0.0
        self._pose_y = 0.0
        self._heading_rad = 0.0
        self._gnss_lost = False
        logger.info("Runtime reset.")

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "gnss_lost": self._gnss_lost,
            "demo_mode": DEMO_MODE,
            "pose_x": self._pose_x,
            "pose_y": self._pose_y,
            "heading_rad": self._heading_rad,
        }


# Singleton runtime — initialised inside lifespan.
runtime = NavDriftRuntime()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        runtime.load()
    except RuntimeError:
        if not DEMO_MODE:
            raise
    yield
    logger.info("Shutting down NavDrift runtime.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NAVDRIFT-0 API",
    version="0.3.0",
    description="Dead-reckoning navigation inference service with WebSocket streaming.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Body-size limit middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": "Request body too large."},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Global exception handler — never leak stack traces
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _check_api_key(key: str) -> bool:
    if not API_KEY:
        return DEMO_MODE  # allow in demo mode even without a key
    return secrets.compare_digest(
        hashlib.sha256(key.encode()).digest(),
        hashlib.sha256(API_KEY.encode()).digest(),
    )


def require_api_key(request: Request) -> None:
    key = request.headers.get("X-API-Key", "")
    if not _check_api_key(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class InitRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    heading_deg: float = Field(..., ge=0, lt=360)
    altitude_m: Optional[float] = None


class SensorReading(BaseModel):
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    baro_hpa: Optional[float] = None
    timestamp_us: Optional[int] = None


class ReacquireRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    heading_deg: float = Field(..., ge=0, lt=360)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.post("/init", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def init_endpoint(request: Request, body: InitRequest):
    """Initialise the navigation runtime with a known fix."""
    runtime.reset()
    runtime.reacquire(body.lat, body.lon, math.radians(body.heading_deg))
    return {"status": "ok", "message": "Runtime initialised."}


@app.post("/ingest", dependencies=[Depends(require_api_key)])
@limiter.limit("120/minute")
async def ingest_endpoint(request: Request, body: SensorReading):
    """Push a single IMU frame and receive the current pose estimate."""
    sensor = body.dict(exclude_none=True)
    try:
        pose = runtime.infer(sensor)
    except Exception:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference failed.")
    return pose


@app.post("/gnss_lost", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def gnss_lost_endpoint(request: Request):
    """Notify the runtime that the GNSS signal has been lost."""
    runtime.mark_gnss_lost()
    return {"status": "ok"}


@app.post("/reacquire", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def reacquire_endpoint(request: Request, body: ReacquireRequest):
    """Provide a new GNSS fix to correct accumulated drift."""
    runtime.reacquire(body.lat, body.lon, math.radians(body.heading_deg))
    return {"status": "ok"}


@app.get("/trajectory", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
async def trajectory_endpoint(request: Request):
    """Return the recorded trajectory points."""
    return {"trajectory": runtime.get_trajectory()}


@app.get("/status", dependencies=[Depends(require_api_key)])
@limiter.limit("120/minute")
async def status_endpoint(request: Request):
    """Return the runtime status."""
    return runtime.status


@app.post("/reset", dependencies=[Depends(require_api_key)])
@limiter.limit("10/minute")
async def reset_endpoint(request: Request):
    """Hard-reset the runtime state."""
    runtime.reset()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WebSocket /ws/stream
# ---------------------------------------------------------------------------
#
# Protocol (client → server):
#   {"type": "sensor",    "data": {<SensorReading fields>}}
#   {"type": "gnss_lost"}
#   {"type": "reacquire", "lat": …, "lon": …, "heading_deg": …}
#
# Protocol (server → client), sent at ~10 Hz:
#   {"type": "pose", "pose_x": …, "pose_y": …, "heading_rad": …,
#    "uncertainty_major": …, "uncertainty_minor": …,
#    "latency_ms": …, "step": …}
#
# Auth: query param ?api_key=<key>  (WS headers unreliable in browsers)
# ---------------------------------------------------------------------------

async def _ws_receive_loop(
    websocket: WebSocket,
    sensor_queue: "asyncio.Queue[Optional[Dict[str, float]]]",
    stop_event: asyncio.Event,
) -> None:
    """
    Receive JSON messages from the client and put sensor readings onto the
    queue.  Sets stop_event when the connection drops.
    """
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON."})
                continue

            msg_type = msg.get("type")

            if msg_type == "sensor":
                data = msg.get("data", {})
                # Validate via Pydantic, drop unknown keys
                try:
                    reading = SensorReading(**data)
                    await sensor_queue.put(reading.dict(exclude_none=True))
                except Exception as exc:
                    await websocket.send_json(
                        {"type": "error", "detail": f"Bad sensor data: {exc}"}
                    )

            elif msg_type == "gnss_lost":
                runtime.mark_gnss_lost()
                await websocket.send_json({"type": "ack", "event": "gnss_lost"})

            elif msg_type == "reacquire":
                try:
                    lat = float(msg["lat"])
                    lon = float(msg["lon"])
                    heading_deg = float(msg.get("heading_deg", msg.get("heading", 0)))
                    runtime.reacquire(lat, lon, math.radians(heading_deg))
                    await websocket.send_json({"type": "ack", "event": "reacquire"})
                except (KeyError, TypeError, ValueError) as exc:
                    await websocket.send_json(
                        {"type": "error", "detail": f"Bad reacquire payload: {exc}"}
                    )

            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unknown message type: {msg_type!r}"}
                )

    except WebSocketDisconnect:
        logger.info("WS client disconnected (receive loop).")
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("WS receive loop error.")
    finally:
        stop_event.set()


async def _ws_inference_loop(
    websocket: WebSocket,
    sensor_queue: "asyncio.Queue[Optional[Dict[str, float]]]",
    stop_event: asyncio.Event,
) -> None:
    """
    Background loop: runs at WS_STREAM_HZ, pulls the latest sensor reading
    from the queue (or reuses the last one), calls runtime.infer(), and
    pushes the pose back to the client.
    """
    interval = 1.0 / WS_STREAM_HZ
    last_sensor: Dict[str, float] = {}  # hold-last on queue empty

    try:
        while not stop_event.is_set():
            step_start = asyncio.get_event_loop().time()

            # Drain the queue; keep only the most recent reading.
            latest: Optional[Dict[str, float]] = None
            while not sensor_queue.empty():
                try:
                    latest = sensor_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if latest is not None:
                last_sensor = latest

            # Run inference (hold-last if no new data yet).
            try:
                pose = runtime.infer(last_sensor)
            except Exception:
                logger.exception("WS inference error at step %d", runtime.status["step"])
                await asyncio.sleep(interval)
                continue

            payload = {"type": "pose", **pose}
            try:
                await websocket.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                # Socket closed while we were sending.
                break

            # Sleep for the remainder of the interval.
            elapsed = asyncio.get_event_loop().time() - step_start
            sleep_for = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep_for)

    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("WS inference loop error.")
    finally:
        stop_event.set()


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket, api_key: str = ""):
    """
    WebSocket endpoint for real-time pose streaming at 10 Hz.

    Authentication is via the `api_key` query parameter, e.g.:
        ws://host/ws/stream?api_key=<key>
    """
    # Authenticate before accepting.
    if not _check_api_key(api_key):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("WS client connected from %s", websocket.client)

    sensor_queue: asyncio.Queue[Optional[Dict[str, float]]] = asyncio.Queue(maxsize=64)
    stop_event = asyncio.Event()

    recv_task = asyncio.create_task(
        _ws_receive_loop(websocket, sensor_queue, stop_event)
    )
    infer_task = asyncio.create_task(
        _ws_inference_loop(websocket, sensor_queue, stop_event)
    )

    # Wait for either task to signal stop (disconnect, error, etc.)
    await stop_event.wait()

    recv_task.cancel()
    infer_task.cancel()
    await asyncio.gather(recv_task, infer_task, return_exceptions=True)

    try:
        await websocket.close()
    except Exception:
        pass  # already closed

    logger.info("WS session ended. Total runtime steps: %d", runtime.status["step"])
