from __future__ import annotations
from dataclasses import dataclass
import math
from orbitforge.core.vector import Vec3
from orbitforge.frames.eci_ecef import eci_to_ecef
from orbitforge.frames.topocentric import ecef_to_enu, az_el_range
from .horizon import HorizonMask

@dataclass(frozen=True)
class GroundStation:
    name: str
    lat_rad: float
    lon_rad: float
    alt_km: float = 0.0
    min_elevation_rad: float = math.radians(5)
    horizon: HorizonMask | None = None

    def elevation_limit(self, az_rad: float) -> float:
        limit = self.min_elevation_rad
        if self.horizon is not None:
            limit = max(limit, self.horizon.elevation_limit(az_rad))
        return limit

def look_angles(station: GroundStation, sat_eci: Vec3, tai_s: float):
    ecef = eci_to_ecef(sat_eci, tai_s)
    enu = ecef_to_enu(ecef, station.lat_rad, station.lon_rad, station.alt_km)
    return az_el_range(enu)

def visible(station: GroundStation, sat_eci: Vec3, tai_s: float) -> bool:
    az, el, _ = look_angles(station, sat_eci, tai_s)
    return el >= station.elevation_limit(az)
