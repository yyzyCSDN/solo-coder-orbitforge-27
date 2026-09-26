from __future__ import annotations
from dataclasses import dataclass
from orbitforge.core.state import TimeWindow
from orbitforge.optimization.golden_section import maximize
from .station import GroundStation, look_angles, elevation_limit

@dataclass(frozen=True)
class PassEvent:
    rise_tai_s: float
    set_tai_s: float
    max_tai_s: float
    max_elevation_rad: float

def _boundary_time(visible_at, t_a: float, t_b: float, tol_s: float=0.01) -> float:
    """Bisect for the visibility transition between two samples.

    visible_at(t_a) and visible_at(t_b) are assumed to differ; the scan
    guarantees this for a rising/setting pair.
    """
    state_a = visible_at(t_a)
    lo, hi = t_a, t_b
    while hi - lo > tol_s:
        mid = 0.5 * (lo + hi)
        if visible_at(mid) == state_a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def _peak(elevation_at, rise: float, set_t: float):
    if set_t - rise <= 1e-9:
        t = 0.5 * (rise + set_t)
        return t, elevation_at(t)
    return maximize(elevation_at, rise, set_t, tolerance=0.5, max_iter=100)

def find_passes(station: GroundStation, position_at, start: float, end: float, step: float=30.0):
    """Find visible passes of the satellite over the station.

    A sample is visible only when its elevation clears both the station's
    minimum elevation and the terrain horizon at its azimuth. Mountain
    ridges that block the satellite mid-pass split one geometric pass into
    separate PassEvents, with rise/set refined to the actual limit
    crossings; each event reports the peak elevation of its own visible
    arc. Blocked arcs narrower than ``step`` may be missed by the scan, so
    callers needing to resolve thin ridges should use a smaller step.
    """
    def visible_at(t):
        az, el, _ = look_angles(station, position_at(t), t)
        return el >= elevation_limit(station, az)

    def elevation_at(t):
        return look_angles(station, position_at(t), t)[1]

    out = []
    inside = False
    rise = None
    prev_t = None
    t = start
    while t <= end + 1e-09:
        now = visible_at(t)
        if now and (not inside):
            rise = _boundary_time(visible_at, prev_t, t) if prev_t is not None else t
            inside = True
        if inside and (not now):
            set_t = _boundary_time(visible_at, prev_t, t)
            peak_t, peak_el = _peak(elevation_at, rise, set_t)
            out.append(PassEvent(rise, set_t, peak_t, peak_el))
            inside = False
        prev_t = t
        t += step
    if inside:
        peak_t, peak_el = _peak(elevation_at, rise, end)
        out.append(PassEvent(rise, end, peak_t, peak_el))
    return out
