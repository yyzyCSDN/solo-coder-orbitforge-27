import math
from orbitforge.core.vector import Vec3
from orbitforge.frames.geodetic import geodetic_to_ecef
from orbitforge.frames.eci_ecef import ecef_to_eci
from orbitforge.visibility.station import GroundStation, look_angles, elevation_limit, visible
from orbitforge.visibility.horizon import HorizonMask
from orbitforge.visibility.passes import find_passes

T0 = 1_700_000_000.0
DURATION = 3600.0
D = math.radians

# Synthetic pass over an equatorial station: smooth arc sweeping azimuth
# east -> north -> west through az=0, peaking at ~56.3 deg elevation.
_STATION_ECEF = geodetic_to_ecef(0.0, 0.0, 0.0)

def _position_at(t):
    s = -1.5 + 3.0 * (t - T0) / DURATION
    enu = Vec3(2000.0 * s, 1000.0, 1500.0 * (1.0 - s * s))
    return ecef_to_eci(_STATION_ECEF + Vec3(enu.z, enu.x, enu.y), t)

def _find(station):
    return find_passes(station, _position_at, T0, T0 + DURATION)

def test_horizon_wrap_interpolation_continuous():
    mask = HorizonMask([(D(350), D(10)), (D(10), D(20)), (D(180), D(5))])
    # az=0 lies on the 350->10 sample arc that straddles the wrap
    assert abs(mask.elevation_limit(0.0) - D(15)) < 1e-12
    # dense scan across 0/360 must show no jump
    prev = mask.elevation_limit(D(359.9))
    for i in range(1, 2001):
        az = D(359.9 + 0.2 * i / 2000.0)
        cur = mask.elevation_limit(az)
        assert abs(cur - prev) < D(0.01)
        prev = cur
    # azimuth normalization is transparent
    assert abs(mask.elevation_limit(D(-10)) - mask.elevation_limit(D(350))) < 1e-12
    assert abs(mask.elevation_limit(D(370)) - mask.elevation_limit(D(10))) < 1e-12

def test_passes_without_terrain_unchanged():
    passes = _find(GroundStation('X', 0, 0))
    assert len(passes) == 1
    p = passes[0]
    # rise/set where elevation crosses the 5 deg minimum, peak ~56.31 deg
    assert abs(p.rise_tai_s - (T0 + 676.7)) < 1.0
    assert abs(p.set_tai_s - (T0 + 2923.3)) < 1.0
    assert abs(p.max_elevation_rad - D(56.31)) < D(0.05)
    assert abs(p.max_tai_s - (T0 + 1800.0)) < 5.0

def test_terrain_wall_delays_rise_only():
    # 30 deg wall over the rising half of the sky (az 180..360)
    wall = HorizonMask([(D(0), 0.0), (D(179.999), 0.0),
                        (D(180), D(30)), (D(359.999), D(30))])
    passes = _find(GroundStation('X', 0, 0, horizon=wall))
    assert len(passes) == 1
    p = passes[0]
    # rise delayed until elevation clears 30 deg; set untouched
    assert abs(p.rise_tai_s - (T0 + 1054.4)) < 1.0
    assert abs(p.set_tai_s - (T0 + 2923.3)) < 1.0
    assert abs(p.max_elevation_rad - D(56.31)) < D(0.05)

def test_terrain_notch_clips_occluded_arc():
    # 53 deg ridge across the north (az 330..30, straddling 0/360)
    notch = HorizonMask([(D(329.999), 0.0), (D(330), D(53)),
                         (D(30), D(53)), (D(30.001), 0.0)])
    station = GroundStation('X', 0, 0, horizon=notch)
    passes = _find(station)
    # occluded arc splits the geometric pass into three visible events
    assert len(passes) == 3
    early, middle, late = passes
    # middle event keeps the true peak; clipped side arcs peak lower
    assert abs(middle.max_elevation_rad - D(56.31)) < D(0.05)
    assert early.max_elevation_rad < D(50.5)
    assert late.max_elevation_rad < D(50.5)
    # events are ordered and separated by the genuinely blocked arcs
    assert early.set_tai_s < middle.rise_tai_s < middle.set_tai_s < late.rise_tai_s
    for t_blocked, t_clear in ((early.set_tai_s, middle.rise_tai_s),
                               (middle.set_tai_s, late.rise_tai_s)):
        mid = 0.5 * (t_blocked + t_clear)
        assert not visible(station, _position_at(mid), mid)
    # every reported event is clear of the terrain at its peak
    for p in passes:
        assert visible(station, _position_at(p.max_tai_s), p.max_tai_s)

def test_refined_boundaries_track_terrain_limit():
    # rolling hills: continuous limit, so every rise/set is an el==limit
    # crossing and the refined boundary times must satisfy it
    hills = HorizonMask([(D(0), D(60)), (D(60), D(8)), (D(120), D(26)),
                         (D(180), D(6)), (D(240), D(30)), (D(300), D(12))])
    station = GroundStation('X', 0, 0, horizon=hills)
    passes = _find(station)
    assert len(passes) >= 2  # hills carve the geometric pass into pieces
    for p in passes:
        for t in (p.rise_tai_s, p.set_tai_s):
            az, el, _ = look_angles(station, _position_at(t), t)
            assert abs(el - elevation_limit(station, az)) < D(0.01)

def test_min_elevation_still_applies_with_terrain():
    # terrain lower than min_elevation must not lower the effective limit
    low = HorizonMask([(D(0), D(1)), (D(180), D(1))])
    station = GroundStation('X', 0, 0, min_elevation_rad=D(5), horizon=low)
    assert abs(elevation_limit(station, D(90)) - D(5)) < 1e-15
    passes = _find(station)
    assert len(passes) == 1
    assert abs(passes[0].rise_tai_s - (T0 + 676.7)) < 1.0
