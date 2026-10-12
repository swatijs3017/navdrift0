"""Smoke tests for NAVDRIFT-0 API (DEMO_MODE)."""
import os
os.environ["DEMO_MODE"] = "true"
os.environ["NAVDRIFT_API_KEY"] = "testkey"

import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)
HEADERS = {"X-API-Key": "testkey"}

def test_status_uninitialized():
    r = client.get("/status", headers=HEADERS)
    # 503 or 200 both acceptable before init
    assert r.status_code in (200, 503)

def test_init():
    r = client.post("/init", json={
        "lat": 28.6139, "lon": 77.2090,
        "heading_deg": 45.0
    }, headers=HEADERS)
    assert r.status_code == 200

def test_ingest():
    client.post("/init", json={
        "lat": 28.6139, "lon": 77.2090,
        "heading_deg": 45.0
    }, headers=HEADERS)
    r = client.post("/ingest", json={
        "accel_x": 0.1, "accel_y": 0.0, "accel_z": 9.8,
        "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.01,
        "speed": 5.0, "steer_angle": 0.0
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "pose_x" in data
    assert "uncertainty_major" in data

def test_gnss_lost():
    r = client.post("/gnss_lost", headers=HEADERS)
    assert r.status_code in (200, 503)

def test_trajectory():
    r = client.get("/trajectory", headers=HEADERS)
    assert r.status_code in (200, 503)

def test_unauthorized():
    r = client.get("/status", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
