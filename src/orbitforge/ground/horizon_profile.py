from __future__ import annotations
import bisect, math

_TWO_PI = 2.0 * math.pi


def _normalized(profile):
    # Wrap azimuths into [0, 2*pi) and merge coincident samples (e.g. 0 and
    # 360 degrees) keeping the higher elevation, so interpolation across
    # north stays continuous instead of jumping or dividing by zero.
    merged = {}
    for az, el in profile:
        key = az % _TWO_PI
        merged[key] = max(merged[key], el) if key in merged else el
    return sorted(merged.items())


def interpolate_profile(profile, azimuth_rad):
    pts = _normalized(profile)
    if not pts:
        return -math.pi / 2.0
    if len(pts) == 1:
        return pts[0][1]
    az = azimuth_rad % _TWO_PI
    azimuths = [a for a, _ in pts]
    i = bisect.bisect_right(azimuths, az)
    a0, e0 = pts[(i - 1) % len(pts)]
    a1, e1 = pts[i % len(pts)]
    if a1 <= a0:
        a1 += _TWO_PI
    if az < a0:
        az += _TWO_PI
    u = (az - a0) / (a1 - a0)
    return e0 + (e1 - e0) * u


def merge_profiles(*profiles):
    azimuths = sorted({a % _TWO_PI for p in profiles for a, _ in p})
    return [(a, max(interpolate_profile(p, a) for p in profiles)) for a in azimuths]


def clearance_margin(profile, azimuth_rad, elevation_rad):
    return elevation_rad - interpolate_profile(profile, azimuth_rad)
