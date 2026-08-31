"""
Trajectory evaluation metrics.

ATE  — Absolute Trajectory Error (metres): global alignment error
RTE  — Relative Trajectory Error (%): drift per unit distance
NLL  — Negative Log-Likelihood under predicted Gaussian
Drift rate — metres of position error per second of GNSS outage
"""

import math
from typing import Optional, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ATE — Absolute Trajectory Error
# ---------------------------------------------------------------------------

def compute_ate(
    gt_poses:   np.ndarray,   # (N, 3) — ground truth [x, y, theta]
    pred_poses: np.ndarray,   # (N, 3) — predicted  [x, y, theta]
    align: bool = True,
) -> float:
    """
    Root-mean-square ATE (position only, in metres).

    If align=True, applies a rigid-body SE(2) alignment (Umeyama method)
    between the two trajectories before computing error — this removes
    initial heading offset but measures shape accuracy.

    Args:
        gt_poses:   (N, 3) ground truth poses
        pred_poses: (N, 3) predicted poses
        align:      whether to align before computing

    Returns:
        RMSE of position errors in metres
    """
    assert gt_poses.shape == pred_poses.shape, "Shape mismatch"

    if align and len(gt_poses) > 2:
        pred_poses = _se2_align(gt_poses[:, :2], pred_poses[:, :2],
                                 pred_full=pred_poses)

    errors = np.linalg.norm(gt_poses[:, :2] - pred_poses[:, :2], axis=-1)
    return float(np.sqrt(np.mean(errors ** 2)))


def _se2_align(
    gt_xy:    np.ndarray,   # (N, 2)
    pred_xy:  np.ndarray,   # (N, 2)
    pred_full: np.ndarray,  # (N, 3)
) -> np.ndarray:
    """
    Least-squares SE(2) alignment of pred onto gt (Umeyama 1991 for 2D).
    Returns the full aligned pred_poses (N, 3).
    """
    mu_gt   = gt_xy.mean(axis=0)
    mu_pred = pred_xy.mean(axis=0)
    gt_c    = gt_xy   - mu_gt
    pred_c  = pred_xy - mu_pred

    W = gt_c.T @ pred_c / len(gt_xy)
    U, _, Vt = np.linalg.svd(W)
    # Ensure a proper rotation (det=1)
    S = np.diag([1.0, np.linalg.det(U @ Vt)])
    R = U @ S @ Vt
    t = mu_gt - R @ mu_pred

    aligned_xy    = (R @ pred_full[:, :2].T).T + t
    aligned_theta = pred_full[:, 2] + math.atan2(R[1, 0], R[0, 0])
    return np.stack([aligned_xy[:, 0], aligned_xy[:, 1], aligned_theta], axis=-1)


# ---------------------------------------------------------------------------
# RTE — Relative Trajectory Error
# ---------------------------------------------------------------------------

def compute_rte(
    gt_poses:   np.ndarray,    # (N, 3)
    pred_poses: np.ndarray,    # (N, 3)
    step_m:     float = 10.0,  # evaluate at every 10m of ground-truth travel
) -> float:
    """
    Relative Trajectory Error as a percentage.

    Computes translation error for fixed-distance sub-sequences and
    returns the mean as a percentage of the sub-sequence length.

    Args:
        gt_poses:   (N, 3) ground truth
        pred_poses: (N, 3) predictions
        step_m:     sub-sequence length in metres

    Returns:
        RTE as % of distance travelled
    """
    if len(gt_poses) < 2:
        return float("nan")

    # Cumulative distances along ground truth
    diffs = np.diff(gt_poses[:, :2], axis=0)
    cum_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=-1))])
    total_dist = cum_dist[-1]
    if total_dist < step_m:
        return float("nan")

    errors: List[float] = []
    start_idx = 0
    while start_idx < len(cum_dist) - 1:
        target_dist = cum_dist[start_idx] + step_m
        end_candidates = np.where(cum_dist >= target_dist)[0]
        if len(end_candidates) == 0:
            break
        end_idx = int(end_candidates[0])

        # Relative transform on ground truth
        gt_start  = gt_poses[start_idx]
        gt_end    = gt_poses[end_idx]
        # Relative transform on prediction
        pred_start = pred_poses[start_idx]
        pred_end   = pred_poses[end_idx]

        # Error in the local frame of start
        def relative_pos_error(start_ref, end_ref, start_est, end_est) -> float:
            theta_ref = start_ref[2]
            c, s = np.cos(-theta_ref), np.sin(-theta_ref)
            R = np.array([[c, -s], [s, c]])
            rel_ref  = R @ (end_ref[:2]  - start_ref[:2])
            rel_est  = R @ (end_est[:2]  - start_est[:2])
            return float(np.linalg.norm(rel_ref - rel_est))

        err = relative_pos_error(gt_start, gt_end, pred_start, pred_end)
        errors.append(err / step_m * 100.0)
        start_idx = end_idx

    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------

def compute_nll(
    mean:    np.ndarray,   # (N, 3)
    cov:     np.ndarray,   # (N, 3, 3)
    targets: np.ndarray,   # (N, 3)
) -> float:
    """
    Mean negative log-likelihood under predicted Gaussians.
    Handles numerical issues gracefully.
    """
    nll_vals = []
    eye = np.eye(3) * 1e-5

    for i in range(len(mean)):
        err = targets[i] - mean[i]
        sigma = cov[i] + eye
        try:
            L = np.linalg.cholesky(sigma)
            log_det = 2.0 * np.log(np.diag(L)).sum()
            v = np.linalg.solve(L, err)
            mahal = float((v * v).sum())
            nll_vals.append(0.5 * (3 * math.log(2 * math.pi) + log_det + mahal))
        except np.linalg.LinAlgError:
            pass

    return float(np.mean(nll_vals)) if nll_vals else float("nan")


# ---------------------------------------------------------------------------
# Drift rate
# ---------------------------------------------------------------------------

def compute_drift_rate(
    gt_poses:     np.ndarray,     # (N, 3)
    pred_poses:   np.ndarray,     # (N, 3)
    outage_duration_s: float,
    imu_hz: float = 100.0,
) -> float:
    """
    Position drift rate in metres per second of GNSS outage.

    Measures the endpoint position error divided by the outage duration.
    """
    if len(gt_poses) == 0 or outage_duration_s <= 0:
        return float("nan")
    endpoint_error = float(np.linalg.norm(gt_poses[-1, :2] - pred_poses[-1, :2]))
    return endpoint_error / outage_duration_s


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def full_evaluation_report(
    gt_poses:          np.ndarray,    # (N, 3)
    pred_poses:        np.ndarray,    # (N, 3)
    pred_mean_deltas:  Optional[np.ndarray] = None,  # (N-1, 3)
    pred_cov:          Optional[np.ndarray] = None,  # (N-1, 3, 3)
    gt_deltas:         Optional[np.ndarray] = None,  # (N-1, 3)
    outage_duration_s: float = 0.0,
) -> Dict[str, float]:
    """Compute the full suite of metrics and return as a dict."""
    report: Dict[str, float] = {}
    report["ate_m"]    = compute_ate(gt_poses, pred_poses, align=True)
    report["rte_pct"]  = compute_rte(gt_poses, pred_poses)
    report["drift_m_per_s"] = compute_drift_rate(gt_poses, pred_poses,
                                                   outage_duration_s)
    if pred_mean_deltas is not None and pred_cov is not None and gt_deltas is not None:
        report["nll"] = compute_nll(pred_mean_deltas, pred_cov, gt_deltas)

    return report


# ---------------------------------------------------------------------------
# Baseline: raw IMU integration (no ML)
# ---------------------------------------------------------------------------

def baseline_imu_integration(
    imu: np.ndarray,          # (N, 6) — [ax, ay, az, gx, gy, gz]
    initial_pose: np.ndarray, # (3,)
    dt: float = 0.01,         # seconds per IMU sample (100Hz default)
) -> np.ndarray:
    """
    Naive dead reckoning by double-integrating accelerometer and integrating gyro.
    This is the worst-case baseline — drifts very fast.

    Returns:
        poses: (N+1, 3)
    """
    poses = [initial_pose.copy()]
    vel   = np.zeros(2)       # 2D velocity estimate

    for i in range(len(imu)):
        ax, ay = imu[i, 0], imu[i, 1]
        gz     = imu[i, 5]   # yaw rate

        theta = poses[-1][2]
        c, s  = np.cos(theta), np.sin(theta)
        # Rotate body-frame accel to world frame
        ax_w = c * ax - s * ay
        ay_w = s * ax + c * ay
        # Integrate
        vel[0] += ax_w * dt
        vel[1] += ay_w * dt
        new_x  = poses[-1][0] + vel[0] * dt
        new_y  = poses[-1][1] + vel[1] * dt
        new_t  = poses[-1][2] + gz * dt
        new_t  = np.arctan2(np.sin(new_t), np.cos(new_t))
        poses.append(np.array([new_x, new_y, new_t]))

    return np.array(poses, dtype=np.float32)


# ---------------------------------------------------------------------------
# Baseline: constant-velocity EKF
# ---------------------------------------------------------------------------

class ConstantVelocityEKF:
    """
    Extended Kalman Filter with a constant-velocity motion model.
    State: [x, y, theta, vx, vy, omega]
    Observation: GPS (x, y, theta) when available.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.x  = np.zeros(6)        # state
        self.P  = np.eye(6) * 0.1    # covariance

        # Process noise
        q = dt ** 2 * 0.1
        self.Q = np.diag([q, q, q * 0.01, q * 10, q * 10, q * 0.1])

        # Observation noise (GPS)
        self.R_gps = np.diag([0.5, 0.5, 0.05])

    def predict(self) -> None:
        dt = self.dt
        F  = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, obs: np.ndarray) -> None:
        """obs: (3,) [x, y, theta]"""
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        innov = obs - H @ self.x
        innov[2] = np.arctan2(np.sin(innov[2]), np.cos(innov[2]))
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_pose(self) -> np.ndarray:
        return self.x[:3].copy()

    def run_sequence(
        self,
        imu: np.ndarray,       # (N, 6) — unused in constant-vel model, structure match
        gps_poses: np.ndarray, # (N, 3) ground truth poses for observation
        gps_mask: np.ndarray,  # (N,) bool — True = GPS available
        initial_pose: np.ndarray,  # (3,)
    ) -> np.ndarray:
        """
        Run the EKF over a sequence.
        Returns predicted poses (N, 3).
        """
        self.x    = np.concatenate([initial_pose, np.zeros(3)])
        self.P    = np.eye(6) * 0.1
        poses_out = []

        for t in range(len(imu)):
            self.predict()
            if gps_mask[t]:
                self.update_gps(gps_poses[t])
            poses_out.append(self.get_pose())

        return np.array(poses_out, dtype=np.float32)
