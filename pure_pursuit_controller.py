import math
from typing import List, Tuple, Optional

Point2 = Tuple[float, float]


class PurePursuitCore:
    def __init__(self, wheelbase: float, lookahead_dist: float, min_seg_len: float = 1e-6):
        self.Lwb = float(wheelbase)
        self.ld = float(lookahead_dist)
        self.min_seg_len = float(min_seg_len)

    def set_lookahead(self, ld: float) -> None:
        self.ld = float(ld)

    def _lookahead_point_along_path(self, path_xy_vehicle: List[Point2]) -> Optional[Point2]:
        if len(path_xy_vehicle) < 2:
            return None

        dist = 0.0
        for i in range(len(path_xy_vehicle) - 1):
            x0, y0 = path_xy_vehicle[i]
            x1, y1 = path_xy_vehicle[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy)

            if seg_len < self.min_seg_len:
                continue

            if dist + seg_len >= self.ld:
                t = (self.ld - dist) / seg_len
                return (x0 + t * dx, y0 + t * dy)
            dist += seg_len

        return path_xy_vehicle[-1]

    def compute(self, path_xy_vehicle: List[Point2]) -> float:
        if len(path_xy_vehicle) < 2:
            return 0.0

        pL = self._lookahead_point_along_path(path_xy_vehicle)
        print(pL,"pl")
        if pL is None:
            return 0.0

        x_t, y_t = pL
        ld2 = x_t * x_t + y_t * y_t
        if ld2 <= 1e-9:
            return 0.0

        kappa = 2.0 * y_t / ld2
        delta = math.atan(self.Lwb * kappa)
        return float(delta)

