import math

G = 9.81
RHO = 1.225


def _derivatives(x, z, vx, vz, wind_x, mass, area, Cd, gravity):
    vx_rel = vx - wind_x
    speed_rel = math.sqrt(vx_rel**2 + vz**2)
    if speed_rel > 1e-9:
        drag = 0.5 * RHO * Cd * area * speed_rel / mass
        return vx, vz, -drag * vx_rel, -drag * vz - gravity
    return vx, vz, 0.0, -gravity


def _rk4(x, z, vx, vz, dt, wind_x, mass, area, Cd, gravity):
    args = (wind_x, mass, area, Cd, gravity)
    def add(s, ds, h): return tuple(a + b * h for a, b in zip(s, ds))
    s = (x, z, vx, vz)
    k1 = _derivatives(*s, *args)
    k2 = _derivatives(*add(s, k1, dt / 2), *args)
    k3 = _derivatives(*add(s, k2, dt / 2), *args)
    k4 = _derivatives(*add(s, k3, dt), *args)
    return tuple(a + (dt / 6) * (p + 2*q + 2*r + e) for a, p, q, r, e in zip(s, k1, k2, k3, k4))


def simulate(speed, elevation_deg, wind_x, mass, radius, Cd, dt=0.01, max_time=60.0, ground_z=0.0, gravity=G):
    """
    Run simulation, return list of (t, x, z). Last point is the landing point.
    ground_z: height of landing platform relative to launch (m). Positive = above, negative = below.
    gravity: gravitational acceleration (m/s²). Defaults to standard Earth gravity (9.81).

    Careful: the last point is only a real landing point if the shot landed
    within max_time -- nothing in the return value says whether it did. Use
    simulate_landing if the endpoint is treated as a measurement.
    """
    traj, _landed = simulate_landing(speed, elevation_deg, wind_x, mass, radius, Cd,
                                     dt=dt, max_time=max_time, ground_z=ground_z,
                                     gravity=gravity)
    return traj


def simulate_landing(speed, elevation_deg, wind_x, mass, radius, Cd, dt=0.01, max_time=60.0,
                     ground_z=0.0, gravity=G):
    """Same integration as simulate, returning `(traj, landed)`.

    `landed` is True only if the trajectory crossed the landing height inside
    max_time. When False, the endpoint is meaningless and must not be stored
    as landing_distance_m.
    """
    area = math.pi * radius**2
    el = math.radians(elevation_deg)
    x, z, vx, vz = 0.0, 0.0, speed * math.cos(el), speed * math.sin(el)
    t = 0.0
    traj = [(t, x, z)]

    while t <= max_time:
        nx, nz, nvx, nvz = _rk4(x, z, vx, vz, dt, wind_x, mass, area, Cd, gravity)
        if nz <= ground_z < z:
            frac = (z - ground_z) / (z - nz)
            traj.append((t + frac * dt, x + frac * (nx - x), ground_z))
            break
        if ground_z > 0 and nz <= 0 < z:
            frac = z / (z - nz)
            traj.append((t + frac * dt, x + frac * (nx - x), 0.0))
            break
        x, z, vx, vz = nx, nz, nvx, nvz
        t += dt
        traj.append((t, x, z))
    else:
        # for/while-else: runs only if the loop wasn't broken out of (no landing).
        return traj, False

    return traj, True
