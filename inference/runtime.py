"""
NavDriftRuntime — production inference engine for NAVDRIFT-0.

DEMO_MODE=true  → EKF simulation, no ONNX needed (Render free tier boots instantly)
DEMO_MODE=false → loads ONNX from HuggingFace Hub or disk
"""
from __future__ import annotations
import logging, os, time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
_ort_ok = False
try:
    import onnxruntime as ort; _ort_ok = True
except ImportError:
    pass

def _wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def _enu(lat, lon, rlat, rlon):
    R=6_371_000.0
    return (R*np.radians(lon-rlon)*np.cos(np.radians(rlat)),
            R*np.radians(lat-rlat))

class _EKF:
    def __init__(self, dt=0.01):
        self.dt=dt; self.x=np.zeros(3); self.P=np.diag([1.,1.,.1])
        self.Q=np.diag([0.01,0.01,0.001]); self.R=np.diag([0.5,0.5,0.02])
    def predict(self, speed, gz):
        dt,th=self.dt,self.x[2]
        self.x[0]+=speed*np.cos(th)*dt; self.x[1]+=speed*np.sin(th)*dt
        self.x[2]=_wrap(self.x[2]+gz*dt)
        F=np.eye(3); F[0,2]=-speed*np.sin(th)*dt; F[1,2]=speed*np.cos(th)*dt
        self.P=F@self.P@F.T+self.Q
    def update(self, ox,oy,ot):
        y=np.array([ox-self.x[0],oy-self.x[1],_wrap(ot-self.x[2])])
        S=self.P+self.R; K=self.P@np.linalg.inv(S)
        self.x+=K@y; self.x[2]=_wrap(self.x[2]); self.P=(np.eye(3)-K)@self.P
    @property
    def unc(self):
        v=np.linalg.eigvalsh(self.P[:2,:2])
        return float(np.sqrt(max(v))), float(np.sqrt(min(v)+1e-9))

class _SNAP:
    def __init__(self, n=15, lr=0.15): self.n=n; self.lr=lr
    def correct(self, traj, tx, ty, tt):
        if not len(traj): return traj
        t=traj.copy().astype(np.float64); N=len(t)
        w=np.linspace(0.,1.,N)
        for _ in range(self.n):
            ex,ey,et=t[-1,0]-tx,t[-1,1]-ty,_wrap(t[-1,2]-tt)
            t[:,0]-=self.lr*w*ex; t[:,1]-=self.lr*w*ey; t[:,2]-=self.lr*w*et
            t[:,2]=np.vectorize(_wrap)(t[:,2])
            if abs(ex)<.05 and abs(ey)<.05 and abs(et)<.005: break
        return t

class NavDriftRuntime:
    def __init__(self, onnx_path="./checkpoints/onnx/drift_former.onnx",
                 norm_stats_path="./checkpoints/drift_former/norm_stats.npz",
                 window=200, imu_hz=100.0):
        self.window=window; self.imu_hz=imu_hz; self.dt=1./imu_hz
        self.ref_lat=self.ref_lon=None; self.gnss_active=False; self.step_count=0
        self.pose_history:Deque[np.ndarray]=deque(maxlen=10_000)
        self.raw_dr_history:Deque[np.ndarray]=deque(maxlen=10_000)
        self._ekf=_EKF(dt=self.dt); self._snap=_SNAP()
        self._buf:Deque[np.ndarray]=deque(maxlen=window)
        self._sess=self._nmean=self._nstd=None
        if not DEMO_MODE: self._load(onnx_path, norm_stats_path)
        else: logger.info("NavDriftRuntime: DEMO_MODE")

    def _load(self, op, np_):
        if not _ort_ok: return
        hf=os.environ.get("HF_REPO_ID","")
        if hf and not os.path.exists(op):
            try:
                from huggingface_hub import hf_hub_download
                op=hf_hub_download(repo_id=hf,filename="drift_former.onnx")
                np_=hf_hub_download(repo_id=hf,filename="norm_stats.npz")
            except Exception as e: logger.error("HF dl failed: %s",e); return
        if not os.path.exists(op): return
        opts=ort.SessionOptions(); opts.intra_op_num_threads=2
        self._sess=ort.InferenceSession(op,sess_options=opts)
        if os.path.exists(np_):
            s=np.load(np_); self._nmean=s["mean"].astype(np.float32); self._nstd=s["std"].astype(np.float32)
        logger.info("ONNX loaded: %s",op)

    @property
    def current_pose(self): return self._ekf.x.copy()

    def set_initial_gnss_fix(self,lat,lon,heading_deg,speed_m_s=0.):
        self.ref_lat=lat; self.ref_lon=lon
        self._ekf.x=np.array([0.,0.,np.radians(heading_deg)])
        self._ekf.P=np.diag([1.,1.,.01]); self.gnss_active=True

    def notify_gnss_lost(self): self.gnss_active=False

    def ingest(self,accel_xyz,gyro_xyz,speed,steer_angle,timestamp=None):
        t0=time.perf_counter()
        obs=np.array([*accel_xyz,*gyro_xyz,speed,steer_angle],dtype=np.float32)
        self._buf.append(obs)
        if self._sess and len(self._buf)>=10: self._onnx()
        else: self._ekf.predict(speed,float(gyro_xyz[2]))
        pose=self._ekf.x.copy(); um,un=self._ekf.unc
        self.pose_history.append(pose); self.raw_dr_history.append(pose.copy()); self.step_count+=1
        return {"pose_x":float(pose[0]),"pose_y":float(pose[1]),"heading_rad":float(pose[2]),
                "uncertainty_major":float(um),"uncertainty_minor":float(un),
                "gnss_active":self.gnss_active,"step":self.step_count,
                "latency_ms":(time.perf_counter()-t0)*1000.}

    def reacquire(self,lat,lon,heading_deg):
        t0=time.perf_counter()
        if self.ref_lat is None: raise ValueError("Not initialized")
        tx,ty=_enu(lat,lon,self.ref_lat,self.ref_lon); tt=np.radians(heading_deg)
        err=float(np.hypot(self._ekf.x[0]-tx,self._ekf.x[1]-ty))
        traj=np.array(list(self.pose_history),dtype=np.float64)
        corr=self._snap.correct(traj,tx,ty,tt)
        if len(corr):
            self.pose_history.clear()
            for pt in corr: self.pose_history.append(pt)
            self._ekf.x=corr[-1].copy()
        self._ekf.update(tx,ty,tt); self.gnss_active=True
        return {"corrected_trajectory":corr.tolist(),"endpoint_error_m":err,
                "runtime_ms":(time.perf_counter()-t0)*1000.}

    def get_full_trajectory(self): return list(self.pose_history)

    def _onnx(self):
        try:
            seq=np.array(list(self._buf),dtype=np.float32)
            if self._nmean is not None: seq=(seq-self._nmean)/(self._nstd+1e-8)
            out=self._sess.run(None,{"input":seq[np.newaxis]})[0][0]
            self._ekf.x[0]+=float(out[0]); self._ekf.x[1]+=float(out[1])
            self._ekf.x[2]=_wrap(self._ekf.x[2]+float(out[2]))
        except Exception as e: logger.warning("ONNX fail: %s",e)
