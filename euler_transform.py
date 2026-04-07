import math
from typing import Iterable, List, Tuple

def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float, *, normalize: bool = True):
    if normalize:
        n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n == 0.0:
            return (0.0, 0.0, 0.0)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = max(-1.0, min(1.0, t2))  
    pitch = math.asin(t2)

   
    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)

    return (yaw, pitch, roll)