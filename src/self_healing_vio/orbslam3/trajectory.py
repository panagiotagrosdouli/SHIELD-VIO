from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TrajectoryPose:
    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


def load_tum_trajectory(path: str | Path) -> List[TrajectoryPose]:
    path = Path(path)
    if not path.exists():
        return []

    poses = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        values = line.split()
        if len(values) != 8:
            continue
        poses.append(
            TrajectoryPose(
                timestamp=float(values[0]),
                tx=float(values[1]),
                ty=float(values[2]),
                tz=float(values[3]),
                qx=float(values[4]),
                qy=float(values[5]),
                qz=float(values[6]),
                qw=float(values[7]),
            )
        )
    return poses
