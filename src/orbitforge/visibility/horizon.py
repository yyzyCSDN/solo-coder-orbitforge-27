from __future__ import annotations
import bisect, math

_TWO_PI = 2.0 * math.pi

class HorizonMask:
    """Terrain horizon mask sampled as (azimuth_rad, elevation_rad) pairs.

    Azimuths wrap into [0, 2*pi); samples that coincide after wrapping
    (e.g. 0 and 360 degrees) are merged keeping the higher elevation, so
    interpolation stays continuous across north instead of jumping.
    """

    def __init__(self, points):
        merged = {}
        for az, el in points:
            key = az % _TWO_PI
            merged[key] = max(merged[key], el) if key in merged else el
        pts = sorted(merged.items())
        self.az = [a for a, _ in pts]
        self.el = [e for _, e in pts]

    def elevation_limit(self, az):
        n = len(self.az)
        if n == 0:
            return -math.pi / 2
        if n == 1:
            return self.el[0]
        a = az % _TWO_PI
        i = bisect.bisect_right(self.az, a)
        i0 = (i - 1) % n
        i1 = i % n
        a0 = self.az[i0]
        a1 = self.az[i1]
        if a1 <= a0:
            a1 += _TWO_PI
        if a < a0:
            a += _TWO_PI
        u = (a - a0) / (a1 - a0)
        return self.el[i0] + (self.el[i1] - self.el[i0]) * u

    def clear(self, az, el):
        return el >= self.elevation_limit(az)
