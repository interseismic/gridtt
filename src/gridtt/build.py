"""Build per-station travel-time tables.

Each (station, phase) table is solved with the FACTORED eikonal equation by 2nd-order fast marching
(``eikonalfm.factored_fast_marching``; Treister & Haber 2016), T = |x - x_station| * tau1. Factoring removes the 1/r
source singularity whose error otherwise spreads through the whole domain and degrades plain FMM to ~1st order.

The lattice of each table is shifted so the STATION lies exactly on a node (eikonalfm takes an integer source index);
the box spans the station and the event volume plus a margin. Tables are stored as one flat float32 buffer + meta.json.
By reciprocity a station-centred table is evaluated at event positions."""
from __future__ import annotations
import json, os, time, datetime
import numpy as np

SOLVER = 'eikonalfm.factored_fast_marching, order 2'


def station_lattice(a, b, zs, box, dx, dz, z_top, z_max, margin):
    """Axes of a table whose lattice contains the station (a, b, zs) exactly as a node."""
    xlo, xhi, ylo, yhi = box
    x0 = min(a, xlo) - margin; y0 = min(b, ylo) - margin; z0 = min(zs, z_top) - margin * 0.5
    x0 = a - np.ceil((a - x0) / dx) * dx; y0 = b - np.ceil((b - y0) / dx) * dx; z0 = zs - np.ceil((zs - z0) / dz) * dz
    X = np.arange(x0, max(a, xhi) + margin + dx, dx); Y = np.arange(y0, max(b, yhi) + margin + dx, dx)
    Z = np.arange(z0, z_max + dz, dz)
    idx = (int(round((a - x0) / dx)), int(round((b - y0) / dx)), int(round((zs - z0) / dz)))
    assert abs(X[idx[0]] - a) < 1e-9 and abs(Y[idx[1]] - b) < 1e-9 and abs(Z[idx[2]] - zs) < 1e-9
    return X, Y, Z, idx


def solve_table(c, idx, spacing):
    import eikonalfm
    c = np.ascontiguousarray(c, dtype=np.float64)
    return eikonalfm.distance(c.shape, spacing, idx, indexing='ij') * eikonalfm.factored_fast_marching(c, idx, spacing, 2)


def _one(args):
    (a, b, zs, ph, box, dx, dz, z_top, z_max, margin, vel) = args
    X, Y, Z, idx = station_lattice(a, b, zs, box, dx, dz, z_top, z_max, margin)
    T = solve_table(vel(ph, X, Y, Z), idx, (dx, dx, dz))
    return T.astype(np.float32), dict(nx=len(X), ny=len(Y), nz=len(Z), x0=float(X[0]), y0=float(Y[0]), z0=float(Z[0]))


def radial_lattice(zs, dr, dz, z_top, z_max, margin, rmax):
    """(r, z) axes of a per-station 2-D table; the station is exactly the node (0, kz)."""
    z0 = zs - np.ceil((zs - (min(zs, z_top) - margin * 0.5)) / dz) * dz
    Z = np.arange(z0, z_max + dz, dz)
    R = np.arange(0.0, rmax + margin + dr, dr)
    kz = int(round((zs - z0) / dz))
    assert abs(Z[kz] - zs) < 1e-9
    return R, Z, kz


def _one_radial(args):
    (a, b, zs, ph, dr, dz, z_top, z_max, margin, rmax, vel) = args
    import eikonalfm
    R, Z, kz = radial_lattice(zs, dr, dz, z_top, z_max, margin, rmax)
    col = np.asarray(vel(ph, [a], [b], Z), dtype=np.float64).reshape(-1)   # 1-D media ignore (a, b)
    c = np.ascontiguousarray(np.broadcast_to(col[None, :], (len(R), len(Z))).copy())
    idx = (0, kz)
    T = eikonalfm.distance(c.shape, (dr, dz), idx, indexing='ij') * \
        eikonalfm.factored_fast_marching(c, idx, (dr, dz), 2)
    return T.astype(np.float32), dict(nr=len(R), nz=len(Z), z0=float(Z[0]))


def build_tables_1d(sta_xyz, box, vel, out_dir, dr=0.25, dz=0.1, z_top=-5.0, z_max=60.0, margin=6.0,
                    names=None, phases=(0, 1), n_jobs=1, log=print, extra_meta=None):
    """Axisymmetric tables for a 1-D (laterally uniform) velocity model: one (r, z) half-plane per station and
    phase, solved with the same factored 2nd-order FMM. In a 1-D medium rays stay in the vertical source-receiver
    plane, so this is EXACTLY the 3-D field at 2-3 orders of magnitude less storage; do not use it with a 3-D
    model (the velocity is sampled as a single profile). Arguments mirror build_tables (dr replaces dx)."""
    os.makedirs(out_dir, exist_ok=True)
    sta_xyz = np.asarray(sta_xyz, dtype=np.float64); S = len(sta_xyz)
    names = list(names) if names is not None else [f'STA{i}' for i in range(S)]
    if len(set(names)) != S:
        raise ValueError('duplicate station names; deduplicate the station list before building')
    xlo, xhi, ylo, yhi = map(float, box)
    corners = np.array([[xlo, ylo], [xlo, yhi], [xhi, ylo], [xhi, yhi]])
    jobs = []
    for ph in phases:
        for i in range(S):
            rmax = float(np.hypot(*(corners - sta_xyz[i, :2]).T).max())
            jobs.append((float(sta_xyz[i, 0]), float(sta_xyz[i, 1]), float(sta_xyz[i, 2]), ph,
                         dr, dz, z_top, z_max, margin, rmax, vel))
    t0 = time.time(); meta = []; off = 0
    tmp = os.path.join(out_dir, 'tables.f32.partial')
    with open(tmp, 'wb') as fh:
        pool = None
        if n_jobs > 1:
            from multiprocessing import get_context
            pool = get_context('fork').Pool(n_jobs); it = pool.imap(_one_radial, jobs, chunksize=1)
        else:
            it = map(_one_radial, jobs)
        for k, (T, g) in enumerate(it):
            i = k % S; ph = phases[k // S]
            T.ravel().tofile(fh)
            meta.append(dict(off=off, sta=names[i], phase=int(ph), **g)); off += T.size
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                log(f'  {k + 1}/{len(jobs)} tables, {off * 4 / 1e9:.3f} GB, {time.time() - t0:.0f} s')
        if pool is not None: pool.close(); pool.join()
    os.replace(tmp, os.path.join(out_dir, 'tables.f32'))
    M = dict(format='gridtt-v1', mode='radial2d', n_cells=int(off), h=float(dr), hz=float(dz), n_sta=S,
             phases=list(map(int, phases)), tables=meta, sta_xyz=sta_xyz.tolist(), sta_names=names,
             box=list(map(float, box)), z_top=z_top, z_max=z_max, margin=margin,
             solver=SOLVER + ' (axisymmetric 2-D, 1-D medium)',
             interpolation='Catmull-Rom bicubic in (r, z), reflected across the axis (C1, interpolating)',
             velocity=getattr(vel, 'description', str(vel)),
             created=datetime.datetime.now().isoformat(timespec='seconds'), build_seconds=round(time.time() - t0, 1))
    if extra_meta: M.update(extra_meta)
    json.dump(M, open(os.path.join(out_dir, 'meta.json'), 'w'))
    log(f'{len(meta)} radial tables, {off * 4 / 1e9:.3f} GB -> {out_dir} in {time.time() - t0:.0f} s')
    return M


def build_tables(sta_xyz, box, vel, out_dir, dx=0.5, dz=0.2, z_top=-5.0, z_max=60.0, margin=6.0,
                 names=None, phases=(0, 1), n_jobs=1, log=print, extra_meta=None):
    """sta_xyz: [S,3] km (projected x, y; z = depth, negative above sea level). box: (xmin, xmax, ymin, ymax) of the
    event volume in km. vel: callable vel(phase, X, Y, Z). Writes out_dir/tables.f32 + out_dir/meta.json."""
    os.makedirs(out_dir, exist_ok=True)
    sta_xyz = np.asarray(sta_xyz, dtype=np.float64); S = len(sta_xyz)
    names = list(names) if names is not None else [f'STA{i}' for i in range(S)]
    if len(set(names)) != S:
        raise ValueError('duplicate station names; deduplicate the station list before building')
    jobs = [(float(sta_xyz[i, 0]), float(sta_xyz[i, 1]), float(sta_xyz[i, 2]), ph, tuple(map(float, box)),
             dx, dz, z_top, z_max, margin, vel) for ph in phases for i in range(S)]
    t0 = time.time(); meta = []; off = 0
    tmp = os.path.join(out_dir, 'tables.f32.partial')
    with open(tmp, 'wb') as fh:
        pool = None
        if n_jobs > 1:
            from multiprocessing import get_context
            pool = get_context('fork').Pool(n_jobs); it = pool.imap(_one, jobs, chunksize=1)
        else:
            it = map(_one, jobs)
        for k, (T, g) in enumerate(it):
            i = k % S; ph = phases[k // S]
            T.ravel().tofile(fh)
            meta.append(dict(off=off, sta=names[i], phase=int(ph), **g)); off += T.size
            if (k + 1) % 20 == 0 or k + 1 == len(jobs):
                log(f'  {k + 1}/{len(jobs)} tables, {off * 4 / 1e9:.2f} GB, {time.time() - t0:.0f} s')
        if pool is not None: pool.close(); pool.join()
    os.replace(tmp, os.path.join(out_dir, 'tables.f32'))
    M = dict(format='gridtt-v1', n_cells=int(off), h=float(dx), hz=float(dz), n_sta=S, phases=list(map(int, phases)),
             tables=meta, sta_xyz=sta_xyz.tolist(), sta_names=names, box=list(map(float, box)), z_top=z_top,
             z_max=z_max, margin=margin, solver=SOLVER, interpolation='Catmull-Rom tricubic (C1, interpolating)',
             velocity=getattr(vel, 'description', str(vel)),
             created=datetime.datetime.now().isoformat(timespec='seconds'), build_seconds=round(time.time() - t0, 1))
    if extra_meta: M.update(extra_meta)
    json.dump(M, open(os.path.join(out_dir, 'meta.json'), 'w'))
    log(f'{len(meta)} tables, {off * 4 / 1e9:.2f} GB -> {out_dir} in {time.time() - t0:.0f} s')
    return M
