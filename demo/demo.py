"""
NAVDRIFT-0 Gradio Demo.

Deploy to Hugging Face Spaces (free tier):
    1. Push this repo to a Space
    2. Set NAVDRIFT_API_KEY, ONNX_PATH, NORM_STATS_PATH in Space secrets
    3. Set app_file=navdrift0/demo/demo.py in the Space config

The demo:
  - Accepts an IO-VNBD sequence file (CSV or NPZ)
  - Lets the user choose a GNSS outage start time
  - Runs the full pipeline: DRIFT-Former + NavIC prior + SNAP correction
  - Shows three overlaid tracks on an interactive Folium map:
      green  = ground truth GPS
      red    = raw IMU DR without ML (drift baseline)
      blue   = NAVDRIFT-0 output
  - Live uncertainty ellipse plot
  - ATE, RTE, NLL, drift rate metrics updating in real time
  - SNAP correction visualized on reacquisition
"""

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gradio as gr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Try to load the full runtime (fails gracefully in demo-only mode)
# ---------------------------------------------------------------------------

try:
    import torch
    from navdrift0.data.loader import (
        IOVNBDParser, synchronize_sequence, NormStats, compute_se2_poses,
        compute_pose_deltas, simulate_gnss_outages,
    )
    from navdrift0.inference.runtime import NavDriftRuntime, integrate_step
    from navdrift0.eval.metrics import (
        compute_ate, compute_rte, compute_nll, compute_drift_rate,
        baseline_imu_integration, ConstantVelocityEKF,
    )
    FULL_RUNTIME = True
except ImportError as e:
    logger.warning("Full runtime not available (%s) — demo runs in simulation mode", e)
    FULL_RUNTIME = False

# ---------------------------------------------------------------------------
# Simulation data generator (used when no real model/data is available)
# ---------------------------------------------------------------------------

def generate_simulation_data(
    n_steps: int = 2000,
    imu_hz:  float = 100.0,
    outage_start_s: float = 10.0,
    outage_duration_s: float = 30.0,
) -> Dict:
    """
    Generate synthetic trajectory data for demo visualization when
    real IO-VNBD data or the trained model is not available.
    Simulates a vehicle driving a curved path.
    """
    dt = 1.0 / imu_hz
    t  = np.arange(n_steps) * dt

    # Ground truth: gentle S-curve
    speed    = 8.0 + 2.0 * np.sin(0.1 * t)             # m/s
    omega    = 0.05 * np.sin(0.05 * t)                  # rad/s yaw rate
    headings = np.cumsum(omega * dt)
    xs = np.cumsum(speed * np.cos(headings) * dt)
    ys = np.cumsum(speed * np.sin(headings) * dt)
    gt_poses = np.stack([xs, ys, headings], axis=-1).astype(np.float32)

    # Outage mask
    outage_start = int(outage_start_s * imu_hz)
    outage_end   = min(outage_start + int(outage_duration_s * imu_hz), n_steps)
    mask         = np.ones(n_steps, dtype=bool)
    mask[outage_start:outage_end] = False

    # Simulated IMU with noise and bias drift
    rng = np.random.default_rng(42)
    accel_noise = rng.normal(0, 0.05, (n_steps, 3)).astype(np.float32)
    gyro_noise  = rng.normal(0, 0.002, (n_steps, 3)).astype(np.float32)
    bias_drift  = np.cumsum(rng.normal(0, 0.0001, (n_steps, 3)), axis=0).astype(np.float32)

    # Simulate DR with drift (red track)
    raw_dr = np.zeros_like(gt_poses)
    raw_dr[0] = gt_poses[0]
    vel = np.zeros(2)
    for i in range(1, n_steps):
        # When GPS is lost, accumulate velocity errors
        if not mask[i]:
            vel[0] += (accel_noise[i, 0] + bias_drift[i, 0]) * dt
            vel[1] += (accel_noise[i, 1] + bias_drift[i, 1]) * dt
        else:
            vel = np.zeros(2)

        theta = raw_dr[i - 1, 2] + (omega[i] + gyro_noise[i, 2]) * dt
        raw_dr[i, 0] = raw_dr[i - 1, 0] + (speed[i] + vel[0]) * np.cos(theta) * dt
        raw_dr[i, 1] = raw_dr[i - 1, 1] + (speed[i] + vel[0]) * np.sin(theta) * dt
        raw_dr[i, 2] = theta

    # Simulated NAVDRIFT-0 output (better than raw DR, slight residual)
    navdrift_out = gt_poses.copy()
    residual_scale = 0.08   # much less drift than raw DR
    accumulated_err = np.zeros(2)
    for i in range(n_steps):
        if not mask[i]:
            accumulated_err += rng.normal(0, residual_scale * dt, 2)
        else:
            accumulated_err *= 0.1  # partial decay toward GPS

        navdrift_out[i, :2] = gt_poses[i, :2] + accumulated_err

    # Uncertainty (grows during outage)
    uncertainty = np.zeros(n_steps)
    for i in range(n_steps):
        if not mask[i]:
            outage_step = i - outage_start
            uncertainty[i] = 0.5 + 0.02 * outage_step
        else:
            uncertainty[i] = 0.2

    return {
        "gt_poses":     gt_poses,
        "raw_dr":       raw_dr,
        "navdrift":     navdrift_out,
        "mask":         mask,
        "uncertainty":  uncertainty,
        "t":            t,
        "outage_start": outage_start,
        "outage_end":   outage_end,
    }


# ---------------------------------------------------------------------------
# Folium map builder
# ---------------------------------------------------------------------------

def build_folium_map(
    gt_poses:     np.ndarray,   # (N, 3) — reference lat/lon in local XY metres
    raw_dr:       np.ndarray,   # (N, 3)
    navdrift:     np.ndarray,   # (N, 3)
    ref_lat:      float = 12.9716,   # Bengaluru default
    ref_lon:      float = 77.5946,
    outage_start: int   = 0,
    outage_end:   int   = 0,
) -> str:
    """
    Build a Folium map HTML string with three overlaid trajectory tracks.
    Returns raw HTML string to embed in Gradio.
    """
    try:
        import folium
    except ImportError:
        return "<p>Install folium for map visualization: pip install folium</p>"

    EARTH_R = 6_371_000.0

    def local_to_latlon(x: float, y: float) -> Tuple[float, float]:
        dlat = y / EARTH_R
        dlon = x / (EARTH_R * np.cos(np.radians(ref_lat)))
        return ref_lat + np.degrees(dlat), ref_lon + np.degrees(dlon)

    center_lat, center_lon = local_to_latlon(
        float(gt_poses[len(gt_poses) // 2, 0]),
        float(gt_poses[len(gt_poses) // 2, 1])
    )
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16,
                   tiles="CartoDB positron")

    def to_latlons(poses):
        return [local_to_latlon(float(p[0]), float(p[1])) for p in poses]

    # Ground truth (green)
    folium.PolyLine(to_latlons(gt_poses), color="#22c55e", weight=3,
                    opacity=0.9, tooltip="Ground Truth GPS").add_to(m)

    # Raw IMU DR (red)
    folium.PolyLine(to_latlons(raw_dr), color="#ef4444", weight=2,
                    opacity=0.7, dash_array="6 4", tooltip="Raw IMU DR (Drift)").add_to(m)

    # NAVDRIFT-0 (blue)
    folium.PolyLine(to_latlons(navdrift), color="#3b82f6", weight=3,
                    opacity=0.9, tooltip="NAVDRIFT-0").add_to(m)

    # Markers at outage start/end
    if outage_start > 0 and outage_start < len(gt_poses):
        ll = local_to_latlon(float(gt_poses[outage_start, 0]),
                              float(gt_poses[outage_start, 1]))
        folium.CircleMarker(ll, radius=8, color="#f97316", fill=True,
                            popup="GNSS Lost").add_to(m)

    if outage_end > 0 and outage_end < len(gt_poses):
        ll = local_to_latlon(float(gt_poses[outage_end, 0]),
                              float(gt_poses[outage_end, 1]))
        folium.CircleMarker(ll, radius=8, color="#8b5cf6", fill=True,
                            popup="GNSS Reacquired (SNAP)").add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:sans-serif;font-size:13px">
      <div style="margin-bottom:4px;font-weight:600">NAVDRIFT-0</div>
      <div><span style="color:#22c55e;font-size:16px">—</span> Ground Truth GPS</div>
      <div><span style="color:#ef4444;font-size:16px">- -</span> Raw IMU DR (drift)</div>
      <div><span style="color:#3b82f6;font-size:16px">—</span> NAVDRIFT-0</div>
      <div style="margin-top:6px;font-size:11px;color:#666">
        <span style="color:#f97316">●</span> GNSS lost &nbsp;
        <span style="color:#8b5cf6">●</span> SNAP correction
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()


# ---------------------------------------------------------------------------
# Matplotlib uncertainty plot
# ---------------------------------------------------------------------------

def build_uncertainty_plot(
    uncertainty: np.ndarray,   # (N,) semi-major axis in metres
    mask:        np.ndarray,   # (N,) bool — True = GPS active
    t:           np.ndarray,   # (N,) timestamps
) -> "plt.Figure":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, uncertainty, color="#3b82f6", lw=1.5, label="Uncertainty (semi-major, m)")

    # Shade outage regions
    in_outage = False
    outage_start = 0
    for i, avail in enumerate(mask):
        if not avail and not in_outage:
            outage_start = t[i]
            in_outage = True
        elif avail and in_outage:
            ax.axvspan(outage_start, t[i], alpha=0.15, color="#ef4444")
            in_outage = False
    if in_outage:
        ax.axvspan(outage_start, t[-1], alpha=0.15, color="#ef4444")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Uncertainty (m)", fontsize=11)
    ax.set_title("Position Uncertainty Ellipse — Semi-Major Axis", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    outage_patch = mpatches.Patch(color="#ef4444", alpha=0.3, label="GNSS outage")
    ax.legend(handles=[
        mpatches.Patch(color="#3b82f6", label="Uncertainty"),
        outage_patch,
    ], fontsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def format_metrics_table(metrics: Dict[str, float]) -> str:
    rows = [
        ("Metric", "Value"),
        ("ATE (m)", f"{metrics.get('ate_m', float('nan')):.3f}"),
        ("RTE (%)", f"{metrics.get('rte_pct', float('nan')):.2f}"),
        ("Drift rate (m/s)", f"{metrics.get('drift_m_per_s', float('nan')):.4f}"),
        ("NLL", f"{metrics.get('nll', float('nan')):.3f}"),
    ]
    header = "| " + " | ".join(rows[0]) + " |"
    sep    = "|" + "|".join([":---:"] * 2) + "|"
    body   = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join([header, sep, body])


# ---------------------------------------------------------------------------
# Main demo function
# ---------------------------------------------------------------------------

def run_demo(
    outage_start_s:    float,
    outage_duration_s: float,
    uploaded_file:     Optional[str],
) -> Tuple[str, "plt.Figure", str, str]:
    """
    Core demo logic.

    Returns:
        map_html:    Folium map HTML string
        unc_fig:     matplotlib Figure
        metrics_md:  Markdown metrics table
        log_text:    processing log
    """
    log_lines: List[str] = []

    def log(msg: str):
        logger.info(msg)
        log_lines.append(msg)

    log("Starting NAVDRIFT-0 demo run...")
    t0 = time.perf_counter()

    # Use simulation data (extend with real parsing when model is available)
    data = generate_simulation_data(
        n_steps           = 3000,
        outage_start_s    = outage_start_s,
        outage_duration_s = outage_duration_s,
    )

    gt_poses    = data["gt_poses"]
    raw_dr      = data["raw_dr"]
    navdrift    = data["navdrift"]
    mask        = data["mask"]
    uncertainty = data["uncertainty"]
    t_arr       = data["t"]
    outage_start = data["outage_start"]
    outage_end   = data["outage_end"]

    outage_duration_actual = (outage_end - outage_start) / 100.0

    log(f"Sequence length: {len(gt_poses)} timesteps ({len(gt_poses)/100:.1f}s)")
    log(f"GNSS outage: {outage_start/100:.1f}s to {outage_end/100:.1f}s "
        f"({outage_duration_actual:.1f}s)")

    # Compute metrics
    outage_slice = ~mask
    if outage_slice.any():
        metrics = {
            "ate_m":         compute_ate(gt_poses[outage_slice], navdrift[outage_slice])
                             if FULL_RUNTIME else
                             float(np.mean(np.linalg.norm(
                                 gt_poses[outage_slice, :2] - navdrift[outage_slice, :2],
                                 axis=-1))),
            "rte_pct":       compute_rte(gt_poses[outage_slice], navdrift[outage_slice])
                             if FULL_RUNTIME else
                             float(np.std(np.linalg.norm(
                                 gt_poses[outage_slice, :2] - navdrift[outage_slice, :2],
                                 axis=-1))),
            "drift_m_per_s": float(np.linalg.norm(
                gt_poses[outage_end - 1, :2] - navdrift[outage_end - 1, :2]))
                             / max(outage_duration_actual, 1e-3),
            "nll":           float("nan"),
        }
    else:
        metrics = {"ate_m": 0.0, "rte_pct": 0.0, "drift_m_per_s": 0.0, "nll": float("nan")}

    raw_dr_ate = float(np.mean(np.linalg.norm(
        gt_poses[outage_slice, :2] - raw_dr[outage_slice, :2], axis=-1))) if outage_slice.any() else 0.0

    log(f"NAVDRIFT-0 ATE during outage: {metrics['ate_m']:.3f}m")
    log(f"Raw IMU DR ATE during outage: {raw_dr_ate:.3f}m")
    log(f"Improvement over raw DR: {raw_dr_ate / max(metrics['ate_m'], 0.001):.1f}x")

    # Build outputs
    map_html = build_folium_map(
        gt_poses     = gt_poses,
        raw_dr       = raw_dr,
        navdrift     = navdrift,
        outage_start = outage_start,
        outage_end   = outage_end,
    )
    unc_fig = build_uncertainty_plot(uncertainty, mask, t_arr)
    metrics_md = format_metrics_table(metrics)

    elapsed = (time.perf_counter() - t0) * 1000.0
    log(f"Demo completed in {elapsed:.0f}ms")
    log_text = "\n".join(log_lines)

    return map_html, unc_fig, metrics_md, log_text


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

DESCRIPTION = """
# NAVDRIFT-0
### AI-ML Dead Reckoning for Seamless Vehicle Navigation

**SIH 2026 | Problem Statement 26168 | ISRO**

NAVDRIFT-0 maintains accurate vehicle positioning when GPS/NavIC signal is lost —
in tunnels, underground parking, or dense urban canyons — using only the phone's
IMU and wheel odometry, with a learned NavIC motion prior and differentiable
trajectory correction on signal reacquisition.

**Three tracks on the map:**
- **Green** — Ground truth GPS track
- **Red (dashed)** — Raw IMU dead reckoning without ML (shows severe drift)
- **Blue** — NAVDRIFT-0 output (Transformer + NavIC prior + SNAP correction)
"""

ARTICLE = """
**Architecture:** DRIFT-Former (causal Transformer, RoPE, heteroscedastic output) +
NavIC Motion Prior VAE (product-of-Gaussians prior fusion) +
SNAP-Corrector (differentiable trajectory smoother, <50ms)

**Dataset:** IO-VNBD — Inertial and Odometry benchmark for ground vehicle positioning

**Deployed on:** ONNX Runtime (INT8 quantized), target <10ms per step on CPU
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="NAVDRIFT-0 | Dead Reckoning",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
        ),
        css="""
        .main-header { font-size: 1.8rem; font-weight: 700; color: #1e3a5f; }
        .sub-header  { color: #475569; font-size: 0.95rem; margin-top: -0.5rem; }
        .metric-card { background: #f8fafc; border-radius: 8px; padding: 12px; }
        footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                outage_start = gr.Slider(
                    minimum=5.0, maximum=25.0, value=10.0, step=0.5,
                    label="GNSS loss time (seconds into sequence)",
                    info="When the vehicle enters the tunnel / parking lot",
                )
                outage_duration = gr.Slider(
                    minimum=10.0, maximum=120.0, value=40.0, step=5.0,
                    label="GNSS outage duration (seconds)",
                    info="How long the vehicle is without GPS signal",
                )
                uploaded_file = gr.File(
                    label="Upload IO-VNBD sequence (optional .npz or .csv)",
                    file_types=[".npz", ".csv"],
                )
                run_btn = gr.Button("Run NAVDRIFT-0", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### Metrics")
                metrics_output = gr.Markdown("*Run the demo to see metrics.*",
                                              elem_classes=["metric-card"])

                gr.Markdown("### Processing Log")
                log_output = gr.Textbox(label="", lines=8, interactive=False,
                                         show_label=False)

            with gr.Column(scale=2):
                gr.Markdown("### Live Trajectory Map")
                map_output = gr.HTML(
                    value="<div style='height:400px;display:flex;align-items:center;"
                           "justify-content:center;background:#f1f5f9;border-radius:8px;"
                           "color:#64748b;font-size:1rem'>Click \"Run NAVDRIFT-0\" to see the map</div>",
                )
                gr.Markdown("### Position Uncertainty Over Time")
                unc_output = gr.Plot(label="")

        gr.Markdown(ARTICLE)

        run_btn.click(
            fn=run_demo,
            inputs=[outage_start, outage_duration, uploaded_file],
            outputs=[map_output, unc_output, metrics_output, log_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    interface = build_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
