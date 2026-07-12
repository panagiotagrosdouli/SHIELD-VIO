"""15-dimensional error-state EKF research prototype for p,v,q,b_a,b_g."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from shield_vio.preintegration.imu_preintegrator import skew, so3_exp


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float); q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; q = np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i+1)%3, (i+2)%3
        s = np.sqrt(1 + R[i,i] - R[j,j] - R[k,k]) * 2
        q = np.zeros(4); q[i+1] = 0.25*s; q[0] = (R[k,j]-R[j,k])/s; q[j+1]=(R[j,i]+R[i,j])/s; q[k+1]=(R[k,i]+R[i,k])/s
    return q / np.linalg.norm(q)


@dataclass
class ESKFState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion_wxyz: np.ndarray
    accel_bias_mps2: np.ndarray
    gyro_bias_rps: np.ndarray
    covariance: np.ndarray
    timestamp_s: float = 0.0


class ErrorStateEKF:
    def __init__(self, state: ESKFState | None = None, gravity_mps2: np.ndarray | None = None) -> None:
        self.state = state or ESKFState(np.zeros(3), np.zeros(3), np.array([1.,0.,0.,0.]), np.zeros(3), np.zeros(3), np.eye(15)*1e-3)
        self.gravity = np.array([0.,0.,-9.80665]) if gravity_mps2 is None else np.asarray(gravity_mps2, dtype=float)
        self._stabilize()

    def propagate(self, acceleration_mps2: np.ndarray, angular_velocity_rps: np.ndarray, dt_s: float,
                  accel_noise: float = 0.05, gyro_noise: float = 0.005,
                  accel_bias_rw: float = 1e-4, gyro_bias_rw: float = 1e-5) -> None:
        if dt_s <= 0: raise ValueError("dt_s must be positive")
        s = self.state; R = quat_to_rot(s.quaternion_wxyz)
        a = np.asarray(acceleration_mps2)-s.accel_bias_mps2; w=np.asarray(angular_velocity_rps)-s.gyro_bias_rps
        aw = R @ a + self.gravity
        s.position_m = s.position_m + s.velocity_mps*dt_s + 0.5*aw*dt_s**2
        s.velocity_mps = s.velocity_mps + aw*dt_s
        s.quaternion_wxyz = rot_to_quat(R @ so3_exp(w*dt_s))
        F=np.eye(15); F[0:3,3:6]=np.eye(3)*dt_s; F[3:6,6:9]=-R@skew(a)*dt_s; F[3:6,9:12]=-R*dt_s; F[6:9,12:15]=-np.eye(3)*dt_s
        G=np.zeros((15,12)); G[3:6,0:3]=R*dt_s; G[6:9,3:6]=np.eye(3)*dt_s; G[9:12,6:9]=np.eye(3)*dt_s; G[12:15,9:12]=np.eye(3)*dt_s
        Q=np.diag([accel_noise**2]*3+[gyro_noise**2]*3+[accel_bias_rw**2]*3+[gyro_bias_rw**2]*3)
        s.covariance=F@s.covariance@F.T+G@Q@G.T; s.timestamp_s += dt_s; self._stabilize()

    def update_position(self, position_m: np.ndarray, covariance_m2: np.ndarray) -> float:
        z=np.asarray(position_m,dtype=float); Rm=np.asarray(covariance_m2,dtype=float)
        H=np.zeros((3,15)); H[:,0:3]=np.eye(3); residual=z-self.state.position_m
        S=H@self.state.covariance@H.T+Rm; K=self.state.covariance@H.T@np.linalg.inv(S)
        dx=K@residual; self._inject(dx)
        I=np.eye(15); A=I-K@H; self.state.covariance=A@self.state.covariance@A.T+K@Rm@K.T
        self._stabilize(); return float(residual@np.linalg.solve(S,residual))

    def _inject(self, dx: np.ndarray) -> None:
        s=self.state; s.position_m+=dx[0:3]; s.velocity_mps+=dx[3:6]
        s.quaternion_wxyz=rot_to_quat(quat_to_rot(s.quaternion_wxyz)@so3_exp(dx[6:9]))
        s.accel_bias_mps2+=dx[9:12]; s.gyro_bias_rps+=dx[12:15]

    def reset_from_pose(self, position_m: np.ndarray, quaternion_wxyz: np.ndarray, covariance_scale: float=1e-3) -> None:
        self.state.position_m=np.asarray(position_m,dtype=float); self.state.quaternion_wxyz=np.asarray(quaternion_wxyz,dtype=float)
        self.state.velocity_mps=np.zeros(3); self.state.covariance=np.eye(15)*covariance_scale; self._stabilize()

    def _stabilize(self) -> None:
        s=self.state; s.quaternion_wxyz=np.asarray(s.quaternion_wxyz,dtype=float); s.quaternion_wxyz/=np.linalg.norm(s.quaternion_wxyz)
        P=np.asarray(s.covariance,dtype=float); P=0.5*(P+P.T); mine=float(np.min(np.linalg.eigvalsh(P)))
        if mine < 1e-12: P += np.eye(15)*(1e-12-mine)
        s.covariance=P
