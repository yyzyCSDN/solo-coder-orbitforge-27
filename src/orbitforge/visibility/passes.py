from __future__ import annotations
from dataclasses import dataclass
from orbitforge.optimization.golden_section import maximize
from .station import GroundStation, look_angles

@dataclass(frozen=True)
class PassEvent:
    rise_tai_s: float
    set_tai_s: float
    max_tai_s: float
    max_elevation_rad: float

def _crossing(margin, t0, t1, tolerance):
    # margin(t0) and margin(t1) straddle zero; bisect the boundary time.
    lo, hi = t0, t1
    flo = margin(lo)
    for _ in range(80):
        if hi - lo <= tolerance:
            break
        mid = 0.5 * (lo + hi)
        if (margin(mid) >= 0.0) == (flo >= 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def _pass_event(elevation, rise, set_t):
    if set_t - rise > 1e-9:
        max_t, max_el = maximize(elevation, rise, set_t)
    else:
        max_t, max_el = rise, elevation(rise)
    return PassEvent(rise, set_t, max_t, max_el)

def find_passes(station: GroundStation, position_at, start: float, end: float, step: float=30.0, tolerance: float=1e-3):
    if step <= 0.0:
        raise ValueError('step must be positive')
    if end < start:
        return []

    def margin(t):
        az, el, _ = look_angles(station, position_at(t), t)
        return el - station.elevation_limit(az)

    def elevation(t):
        return look_angles(station, position_at(t), t)[1]

    passes = []
    rise = None
    t0 = start
    m0 = margin(t0)
    if m0 >= 0.0:
        rise = t0
    t1 = start + step
    while True:
        t1 = min(t1, end)
        m1 = margin(t1)
        if m0 < 0.0 and m1 >= 0.0:
            rise = _crossing(margin, t0, t1, tolerance)
        elif m0 >= 0.0 and m1 < 0.0:
            passes.append(_pass_event(elevation, rise, _crossing(margin, t0, t1, tolerance)))
            rise = None
        if t1 >= end:
            break
        t0, m0 = t1, m1
        t1 += step
    if rise is not None:
        passes.append(_pass_event(elevation, rise, end))
    return passes
