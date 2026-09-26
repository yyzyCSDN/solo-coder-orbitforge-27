import math
from orbitforge.time.leap_seconds import utc_unix_to_tai,tai_to_utc_unix
from orbitforge.core.vector import Vec3
from orbitforge.frames.eci_ecef import eci_to_ecef,ecef_to_eci
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.lambert import lambert_universal
from orbitforge.visibility.station import GroundStation,look_angles
from orbitforge.visibility.horizon import HorizonMask
from orbitforge.visibility.passes import find_passes
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.link.budget import free_space_loss_db,received_power_dbw
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.conjunction.covariance import rotate_covariance_2d
from orbitforge.conjunction.probability import collision_probability_2d
from orbitforge.ephemeris.interpolation import EphemerisPoint,hermite
from orbitforge.mission.windows import intersect_windows
from orbitforge.core.state import TimeWindow

def test_b01_leap_roundtrip():
    utc=1483228800.25; tai=utc_unix_to_tai(utc); assert abs(tai-utc-37)<1e-9; assert abs(tai_to_utc_unix(tai)-utc)<1e-9

def test_b02_frame_roundtrip():
    v=Vec3(7000,-1200,900); t=1700000000.0; q=ecef_to_eci(eci_to_ecef(v,t),t); assert (q-v).norm()<1e-9

def test_b03_kepler_high_eccentricity():
    e=.95; m=2.4; E=solve_kepler_elliptic(m,e); assert abs(E-e*math.sin(E)-((m+math.pi)%(2*math.pi)-math.pi))<1e-10

def test_b04_lambert_hits_transfer_direction():
    r1=Vec3(7000,0,0); r2=Vec3(0,8000,0); v1,v2=lambert_universal(r1,r2,1800); assert v1.y>0 and v2.x<0

def test_b05_ground_station_zenith():
    from orbitforge.frames.geodetic import geodetic_to_ecef
    from orbitforge.frames.rotations import rz,mv,transpose
    from orbitforge.time.sidereal import gmst_angle
    lat=math.radians(35); lon=math.radians(139); t=1700000000
    st=GroundStation('X',lat,lon,0,0)
    ecef=geodetic_to_ecef(lat,lon,500)
    sat=mv(transpose(rz(gmst_angle(t))),ecef)
    az,el,r=look_angles(st,sat,t)
    assert el>math.radians(89) and 499<r<501

def test_b06_eclipse_night_side():
    assert eclipse_state(Vec3(-7000,0,0),Vec3(149597870,0,0))=='umbra'
    assert eclipse_state(Vec3(7000,0,0),Vec3(149597870,0,0))=='sunlit'

def test_b07_link_fspl_dimension():
    loss=free_space_loss_db(1000,2e9); assert 158<loss<159
    assert abs(received_power_dbw(10,20,30,1000,2e9,3)-(10+20+30-loss-3))<1e-12

def test_b08_quaternion_rotation():
    q=Quaternion.from_axis_angle(Vec3(0,0,1),math.pi/2); v=q.rotate(Vec3(1,0,0)); assert abs(v.x)<1e-9 and abs(v.y-1)<1e-9

def test_b09_covariance_rotation():
    c=((4.0,0.0),(0.0,1.0)); r=rotate_covariance_2d(c,math.pi/2); assert abs(r[0][0]-1)<1e-9 and abs(r[1][1]-4)<1e-9

def test_b10_collision_probability_monotonic():
    cov=((1.0,0.0),(0.0,1.0)); p1=collision_probability_2d(0,0,cov,.05); p2=collision_probability_2d(4,0,cov,.05); assert p1>p2>0

def test_b11_hermite_endpoints():
    p0=EphemerisPoint(0,Vec3(1,2,3),Vec3(4,5,6)); p1=EphemerisPoint(10,Vec3(11,12,13),Vec3(1,2,3)); assert hermite(p0,p1,0)==p0; assert hermite(p0,p1,10)==p1

def test_b12_window_half_open_semantics():
    a=[TimeWindow(0,10),TimeWindow(20,30)]; b=[TimeWindow(10,20),TimeWindow(25,35)]; out=intersect_windows(a,b); assert len(out)==1 and out[0]==TimeWindow(25,30)

def _arc_position_at(peak_el_deg=30.0,duration_s=1000.0,az_rate_deg_s=0.02):
    from orbitforge.frames.geodetic import geodetic_to_ecef
    from orbitforge.frames.eci_ecef import ecef_to_eci
    origin=geodetic_to_ecef(0.0,0.0,0.0)
    def position_at(t):
        el=math.radians(peak_el_deg)*math.sin(math.pi*t/duration_s)
        az=math.radians(90.0+(t-duration_s/2.0)*az_rate_deg_s)
        enu=Vec3(1000.0*math.cos(el)*math.sin(az),1000.0*math.cos(el)*math.cos(az),1000.0*math.sin(el))
        return ecef_to_eci(origin+Vec3(enu.z,enu.x,enu.y),t)
    return position_at

def test_b13_horizon_mask_interpolates_across_north():
    m=HorizonMask([(math.radians(350),math.radians(10)),(math.radians(10),math.radians(20))])
    assert abs(m.elevation_limit(0.0)-math.radians(15))<1e-9
    assert abs(m.elevation_limit(2*math.pi)-m.elevation_limit(0.0))<1e-12
    prev=m.elevation_limit(0.0); k=1
    while k<=36000:
        e=m.elevation_limit(math.radians(k*0.01)); assert abs(e-prev)<math.radians(0.02); prev=e; k+=1

def test_b14_horizon_mask_merges_duplicate_azimuth():
    m=HorizonMask([(0.0,math.radians(5)),(2*math.pi,math.radians(9)),(math.radians(180),math.radians(30))])
    assert abs(m.elevation_limit(0.0)-math.radians(9))<1e-12
    assert abs(m.elevation_limit(math.radians(359.999))-m.elevation_limit(math.radians(0.001)))<math.radians(0.01)

def test_b15_horizon_profile_wrap_and_duplicates():
    from orbitforge.ground.horizon_profile import interpolate_profile
    p=[(0.0,math.radians(5)),(2*math.pi,math.radians(9)),(math.radians(180),math.radians(30))]
    assert abs(interpolate_profile(p,0.0)-math.radians(9))<1e-12
    assert abs(interpolate_profile(p,math.radians(360))-interpolate_profile(p,0.0))<1e-12
    assert abs(interpolate_profile(p,math.radians(-90))-interpolate_profile(p,math.radians(270)))<1e-12

def test_b16_pass_terrain_shifts_rise_and_set():
    pos=_arc_position_at()
    flat=GroundStation('F',0.0,0.0,0.0,0.0)
    base=find_passes(flat,pos,0.0,1000.0,step=10.0)
    assert len(base)==1 and abs(base[0].rise_tai_s)<1e-6 and abs(base[0].set_tai_s-1000.0)<1e-6
    assert abs(base[0].max_tai_s-500.0)<0.5 and abs(base[0].max_elevation_rad-math.radians(30))<math.radians(0.01)
    mask=HorizonMask([(math.radians(a),math.radians(10)) for a in range(0,360,15)])
    st=GroundStation('M',0.0,0.0,0.0,0.0,mask)
    out=find_passes(st,pos,0.0,1000.0,step=10.0)
    crossing=1000.0/math.pi*math.asin(1.0/3.0)
    assert len(out)==1 and abs(out[0].rise_tai_s-crossing)<0.5 and abs(out[0].set_tai_s-(1000.0-crossing))<0.5
    assert abs(out[0].max_elevation_rad-math.radians(30))<math.radians(0.01)

def test_b17_pass_terrain_peak_occlusion_splits_pass():
    pos=_arc_position_at()
    pts=[(math.radians(80),0.0),(math.radians(88),0.0),(math.radians(90),math.radians(40)),(math.radians(92),0.0),(math.radians(100),0.0)]
    st=GroundStation('P',0.0,0.0,0.0,0.0,HorizonMask(pts))
    out=find_passes(st,pos,0.0,1000.0,step=10.0)
    assert len(out)==2
    assert abs(out[0].rise_tai_s)<1e-6 and 470.0<out[0].set_tai_s<480.0
    assert 520.0<out[1].rise_tai_s<530.0 and abs(out[1].set_tai_s-1000.0)<1e-6
    assert math.radians(29.8)<out[0].max_elevation_rad<math.radians(30.0)
    assert math.radians(29.8)<out[1].max_elevation_rad<math.radians(30.0)

def test_b18_pass_fully_occluded_by_terrain():
    pos=_arc_position_at()
    mask=HorizonMask([(math.radians(a),math.radians(35)) for a in range(0,360,30)])
    st=GroundStation('B',0.0,0.0,0.0,0.0,mask)
    assert find_passes(st,pos,0.0,1000.0,step=10.0)==[]

def test_b19_pass_min_elevation_dominates_low_horizon():
    pos=_arc_position_at()
    mask=HorizonMask([(math.radians(a),math.radians(5)) for a in range(0,360,15)])
    st=GroundStation('L',0.0,0.0,0.0,math.radians(10),mask)
    out=find_passes(st,pos,0.0,1000.0,step=10.0)
    crossing=1000.0/math.pi*math.asin(1.0/3.0)
    assert len(out)==1 and abs(out[0].rise_tai_s-crossing)<0.5 and abs(out[0].set_tai_s-(1000.0-crossing))<0.5
