"""Self-check: is the operator accurate for *your* velocity model and *your* area?

`validate_tables` rebuilds a spread of stations at a refined spacing (default: half in every axis) and compares the
production tables against that reference at points covering the whole domain -- travel time, the gradients autograd
actually uses, and a high-frequency-artefact score on vertical profiles. It is model- and region-agnostic: everything
it needs (box, stations, spacing, velocity file) comes from the table directory's meta.json.
"""
from __future__ import annotations
import json, os, shutil, tempfile
import numpy as np, torch

from .build import build_tables
from .operator import load_gridtt


def _spread(xy, n, seed=0):
    """n station indices spread over the network (k-means++ centroids -> nearest real station)."""
    from scipy.cluster.vq import kmeans2
    n = min(n, len(xy))
    cen, _ = kmeans2(np.asarray(xy, float), n, minit='++', seed=seed)
    return sorted({int(np.argmin(((xy - c) ** 2).sum(1))) for c in cen})


def _hp_rms(v, dz, win=2.0):
    """RMS of v after removing a running mean of length `win` km: structure the physics cannot have."""
    k = max(3, int(round(win / dz)) | 1)
    sm = np.convolve(np.pad(v, k // 2, mode='reflect'), np.ones(k) / k, mode='valid')
    return float(np.sqrt(np.mean((v - sm) ** 2)))


def _eval(model, X):
    x = torch.tensor(np.asarray(X, np.float32), requires_grad=True)
    T = model(x).squeeze(1)
    g, = torch.autograd.grad(T.sum(), x)
    return T.detach().numpy(), g[:, :3].numpy()


def validate_tables(table_dir: str, velocity=None, refine: int = 2, n_sta: int = 6, n_points: int = 200_000,
                    events=None, ref_dir=None, keep_ref: bool = False, n_jobs: int = 1, gate_hf: float = 0.01,
                    min_range=None, seed: int = 0, out_json: str | None = None, log=print) -> dict:
    """Compare `table_dir` with a reference built at `refine`x finer spacing.

    velocity : the same velocity object used to build the tables (a `LayeredGridVelocity`, `Profile1D`,
               `FunctionVelocity` or any callable ``vel(phase, X, Y, Z)``). If None, it is rebuilt from
               `meta['velocity_file']`/`meta['velocity_format']` when those were recorded at build time.
    events    : optional array [N,3] of real source points (projected km) to use instead of a uniform sample.
    gate_hf   : pass/fail threshold for the high-frequency artefact score, as a FRACTION of the medium's
                slowness (median |grad T| = 1/v), so it means the same thing in any model, region or unit
                system. 0.01 = the artefact must stay below 1 % of 1/v.
    min_range : source-receiver distances below this are reported separately, not pooled: at r -> 0 the exact
                gradient is discontinuous and no interpolant (or network) is accurate there. Default 2 cells.
    Returns a dict of statistics; also written to `out_json` when given.
    """
    meta = json.load(open(os.path.join(table_dir, 'meta.json')))
    h, hz = float(meta['h']), float(meta.get('hz', meta['h']))
    sta = np.array(meta['sta_xyz'], float); names = meta.get('sta_names')
    if velocity is None:
        velocity = _velocity_from_meta(meta)
    pick = _spread(sta[:, :2], n_sta, seed)
    log(f'validating {table_dir}: dx {h} dz {hz} km, {len(sta)} stations, testing {len(pick)} of them '
        f'against a {refine}x finer reference')

    # volume every selected table covers (avoids edge clamping in the comparison)
    if str(meta.get('mode', 'grid3d')) == 'radial2d':
        # every radial table spans the event box (+margin) in map view and [z_top, z_max] in depth
        box = [float(v) for v in meta['box']]
        lo = np.array([box[0] + h, box[2] + h, float(meta['z_top']) + hz])
        hi = np.array([box[1] - h, box[3] - h, float(meta['z_max']) - hz])
    else:
        T = {(t['sta'] if names else k % len(sta), t['phase']): t for k, t in enumerate(meta['tables'])}
        sel = [names[i] if names else i for i in pick]
        tb = [T[(s, 0)] for s in sel]
        lo = np.array([[t['x0'], t['y0'], t['z0']] for t in tb]).max(0) + np.array([h, h, hz])
        hi = np.array([[t['x0'] + (t['nx'] - 1) * h, t['y0'] + (t['ny'] - 1) * h, t['z0'] + (t['nz'] - 1) * hz]
                       for t in tb]).min(0) - np.array([h, h, hz])
    log(f'  common volume  x {lo[0]:.1f}..{hi[0]:.1f}  y {lo[1]:.1f}..{hi[1]:.1f}  z {lo[2]:.1f}..{hi[2]:.1f} km')

    own_tmp = ref_dir is None
    ref_dir = ref_dir or tempfile.mkdtemp(prefix='gridtt_ref_')
    try:
        return _compare(table_dir, ref_dir, meta, velocity, refine, pick, names, sta, lo, hi, h, hz, n_points,
                        events, n_jobs, gate_hf, min_range, seed, out_json, log)
    finally:
        if own_tmp and not keep_ref:
            shutil.rmtree(ref_dir, ignore_errors=True)


def _compare(table_dir, ref_dir, meta, velocity, refine, pick, names, sta, lo, hi, h, hz, n_points, events,
             n_jobs, gate_hf, min_range, seed, out_json, log):
    if not os.path.exists(os.path.join(ref_dir, 'meta.json')):
        box = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))
        common = dict(z_top=float(meta.get('z_top', lo[2])), z_max=float(hi[2]),
                      margin=float(meta.get('margin', 6.0)),
                      names=[names[i] for i in pick] if names else None,
                      phases=tuple(meta.get('phases', (0, 1))), n_jobs=n_jobs, log=log,
                      extra_meta={'purpose': f'reference for {os.path.abspath(table_dir)}'})
        if str(meta.get('mode', 'grid3d')) == 'radial2d':
            from .build import build_tables_1d
            build_tables_1d(sta[pick], box, velocity, ref_dir, dr=h / refine, dz=hz / refine, **common)
        else:
            build_tables(sta[pick], box, velocity, ref_dir, dx=h / refine, dz=hz / refine, **common)
    A = load_gridtt(table_dir, device='cpu'); B = load_gridtt(ref_dir, device='cpu')

    rng = np.random.default_rng(seed)
    phases = list(meta.get('phases', (0, 1)))
    rows = []
    for i in pick:
        for ph in phases:
            n = max(2000, n_points // (len(pick) * len(phases)))
            if events is not None and len(events):
                e = np.asarray(events, float)
                e = e[(e[:, 0] > lo[0]) & (e[:, 0] < hi[0]) & (e[:, 1] > lo[1]) & (e[:, 1] < hi[1]) &
                      (e[:, 2] > lo[2]) & (e[:, 2] < hi[2])]
                src = e[rng.choice(len(e), n)] if len(e) else None
            else:
                src = None
            if src is None:
                src = rng.uniform(lo, hi, size=(n, 3))
            rows.append(np.column_stack([src, np.tile(sta[i], (n, 1)), np.full(n, ph)]))
    X = np.concatenate(rows).astype(np.float32)
    Ta, Ga = _eval(A, X); Tb, Gb = _eval(B, X)
    rng_km = np.linalg.norm(X[:, :3] - X[:, 3:6], axis=1)
    rmin = float(min_range if min_range is not None else 2 * max(h, hz))
    far = rng_km >= rmin
    near_frac = float(1.0 - far.mean())
    dT = np.abs(Ta - Tb)[far] * 1e3
    dgz = np.abs(Ga[far, 2] - Gb[far, 2]) * 1e3
    dgh = np.hypot(Ga[far, 0] - Gb[far, 0], Ga[far, 1] - Gb[far, 1]) * 1e3
    # The eikonal gives the scale for free: |grad T| = 1/v exactly. Using the full gradient norm (not its
    # vertical component) makes the reference slowness a property of the medium, not of the ray geometry.
    slowness = float(np.median(np.linalg.norm(Gb[far], axis=1))) * 1e3      # ms/km

    # high-frequency artefact score on vertical profiles covering the map
    zp = np.arange(float(lo[2]), float(hi[2]), min(0.05, hz / 2), dtype=np.float32)
    gx = np.linspace(lo[0] + h, hi[0] - h, 6); gy = np.linspace(lo[1] + h, hi[1] - h, 6)
    scores, pmax = [], []
    for i in pick:
        for px in gx:
            for py in gy:
                P = np.column_stack([np.full_like(zp, px), np.full_like(zp, py), zp,
                                     np.tile(sta[i], (len(zp), 1)), np.zeros_like(zp)]).astype(np.float32)
                keep = np.linalg.norm(P[:, :3] - P[:, 3:6], axis=1) >= rmin
                if keep.sum() < 20:
                    continue
                _, ga = _eval(A, P); _, gb = _eval(B, P)
                e = ((ga[:, 2] - gb[:, 2]) * 1e3)[keep]
                scores.append(_hp_rms(e, float(zp[1] - zp[0]))); pmax.append(float(np.abs(e).max()))
    q = lambda v: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                       p99=float(np.percentile(v, 99)), max=float(np.max(v)))
    hf = np.array(scores)
    # gate on the p90 profile, not the max: the worst profiles sit on velocity contrasts, where the exact dT/dz
    # has a corner and the error is real, local and bounded (reported as hf_artefact_ms_per_km['max']).
    hf_rel = float(np.percentile(hf, 90) / slowness)
    out = dict(table_dir=os.path.abspath(table_dir), dx=h, dz=hz, refine=refine, n_pairs=int(far.sum()),
               stations=[names[i] if names else i for i in pick],
               min_range_km=rmin, near_source_fraction_excluded=near_frac, median_slowness_ms_per_km=slowness,
               travel_time_ms=q(dT), dTdz_ms_per_km=q(dgz), dTdx_ms_per_km=q(dgh),
               dTdz_relative=dict(median=float(np.median(dgz) / slowness), p99=float(np.percentile(dgz, 99) / slowness)),
               hf_artefact_ms_per_km=q(hf), hf_artefact_relative_p90=hf_rel,
               hf_artefact_relative_max=float(np.max(hf) / slowness), hf_gate_relative=gate_hf,
               n_profiles=len(hf), profile_max_error_ms_per_km=float(np.max(pmax)),
               passed=bool(hf_rel < gate_hf))
    log(f'  pooled over {out["n_pairs"]:,} pairs with range >= {rmin:g} km '
        f'({near_frac*100:.1f} % nearer than that, excluded); medium slowness |grad T| = {slowness:.0f} ms/km '
        f'(v = {1e3/slowness:.2f} km/s)')
    log(f'  |dT|        median {out["travel_time_ms"]["median"]:.3f} ms   p99 {out["travel_time_ms"]["p99"]:.3f}   '
        f'max {out["travel_time_ms"]["max"]:.3f}')
    log(f'  |d dT/dz|   median {out["dTdz_ms_per_km"]["median"]:.3f} ms/km   p99 {out["dTdz_ms_per_km"]["p99"]:.3f}   '
        f'max {out["dTdz_ms_per_km"]["max"]:.3f}')
    log(f'  |d dT/dx|   median {out["dTdx_ms_per_km"]["median"]:.3f} ms/km   p99 {out["dTdx_ms_per_km"]["p99"]:.3f}')
    log(f'  HF artefact over {len(hf)} profiles: median {out["hf_artefact_ms_per_km"]["median"]:.3f}, '
        f'p90 {np.percentile(hf, 90):.3f}, max {out["hf_artefact_ms_per_km"]["max"]:.3f} ms/km '
        f'(worst profiles sit on velocity contrasts)')
    log(f'  p90 = {hf_rel*100:.3f} % of the {slowness:.0f} ms/km slowness (gate {gate_hf*100:g} %) -> '
        f'{"PASS" if out["passed"] else "FAIL: refine dz (or dx)"}')
    if out_json:
        json.dump(out, open(out_json, 'w'), indent=1); log(f'  wrote {out_json}')
    return out


def _velocity_from_meta(meta: dict):
    from pyproj import Proj
    from .velocity import open_velocity
    vf = meta.get('velocity_file')
    if not vf or not os.path.exists(vf):
        raise ValueError('velocity model not recorded in meta.json (or file moved); pass velocity=...')
    proj = None
    if 'lon0' in meta and 'lat0' in meta:
        proj = Proj(proj='laea', lon_0=float(meta['lon0']), lat_0=float(meta['lat0']), datum='WGS84', units='km')
    return open_velocity(vf, proj, meta.get('velocity_format', 'auto'))
