"""Velocity models for GridTT.

The velocity model is interpolated CONSISTENTLY -- linearly (Delaunay) in map view within each depth level and linearly
in depth between levels -- so that tables built at any grid spacing see the same continuous medium. (Snapping to the
nearest node makes every spacing a different piecewise-constant medium and destroys grid convergence.)"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class LayeredGridVelocity:
    """Vp/Vs given on scattered map nodes at a discrete set of depth levels (a 3-D tomography model).

    Callable as ``vel(phase, X, Y, Z) -> array[len(X), len(Y), len(Z)]`` in km/s, with X, Y in the projected km frame
    and Z depth in km (positive down). phase 0 = P, 1 = S."""

    def __init__(self, x_km, y_km, depth_km, vp, vs, description: str = ""):
        self.levels = np.sort(np.unique(depth_km))
        self.description = description
        self._lev = {0: [], 1: []}
        for z in self.levels:
            m = depth_km == z; xy = np.c_[x_km[m], y_km[m]]
            for ph, v in ((0, vp[m]), (1, vs[m])):
                self._lev[ph].append((LinearNDInterpolator(xy, v), NearestNDInterpolator(xy, v)))

    @classmethod
    def from_ascii8(cls, path: str, proj):
        """Headerless 8-column whitespace ASCII, as several regional tomography models are distributed:
        ``lon lat z[m] . . tag vp[m/s] vs[m/s]`` (columns 4, 5 and 6 are ignored)."""
        V = pd.read_csv(path, sep=r'\s+', header=None, names=['lon', 'lat', 'zm', 'a', 'b', 'tag', 'vp', 'vs'])
        x, y = proj(V.lon.values, V.lat.values)
        return cls(np.asarray(x), np.asarray(y), V.zm.values / 1000.0, V.vp.values / 1000.0, V.vs.values / 1000.0,
                   description=f'ascii8:{path}')

    @classmethod
    def from_csv(cls, path: str, proj, lon=None, lat=None, depth=None, vp=None, vs=None, x=None, y=None):
        """Generic CSV of Vp (and Vs or Vp/Vs) on depth levels, for any model and any region.

        Column names are matched case-insensitively from the usual aliases (lon/longitude/x, lat/latitude/y,
        depth/depth_km/z, vp/vel_p, vs/vel_s, vpvs/vp_vs); units are detected (m/s vs km/s, m vs km, elevation
        vs depth) -- see `_columns`. Give explicit names to override. If the file carries projected km columns
        (x, y) instead of lon/lat, no projection is needed."""
        V = pd.read_csv(path)
        X, Y, Z, P, S = _columns(V, proj, lon=lon, lat=lat, depth=depth, vp=vp, vs=vs, x=x, y=y)
        return cls(X, Y, Z, P, S, description=f'csv:{path}')

    def __call__(self, phase: int, X, Y, Z):
        gx, gy = np.meshgrid(X, Y, indexing='ij'); pts = np.c_[gx.ravel(), gy.ravel()]
        lv = np.empty((len(X), len(Y), len(self.levels)))
        for k, (li, ni) in enumerate(self._lev[phase]):
            v = li(pts); bad = ~np.isfinite(v)
            if bad.any(): v[bad] = ni(pts[bad])
            lv[:, :, k] = v.reshape(len(X), len(Y))
        L = self.levels
        zc = np.clip(Z, L[0], L[-1]); j = np.clip(np.searchsorted(L, zc) - 1, 0, len(L) - 2)
        w = (zc - L[j]) / (L[j + 1] - L[j])
        return np.ascontiguousarray(lv[:, :, j] * (1 - w) + lv[:, :, j + 1] * w)


class LayerCake1D:
    """Laterally uniform model given as LAYER TOPS: constant velocity (or a gradient) inside each layer.

    This is the NLLoc/HYPOINVERSE convention -- a row is the top of a layer, and the velocity applies from that
    depth down to the next row. It is NOT the same medium as `Profile1D`, which interpolates between rows; read a
    layer list the wrong way and the velocity at mid-layer can be off by tens of percent.

    Every layer top is a true velocity discontinuity, which no grid represents exactly. `smooth_km` replaces each
    jump with a linear ramp of that full width, centred on the layer top: set it to about two cells (2 x dz) and
    the model becomes representable, at the cost of a transition zone that thin. `smooth_km = 0` (the default)
    keeps the discontinuity exactly as given."""

    def __init__(self, path: str, depth=None, vp=None, vs=None, smooth_km: float = 0.0, eps: float = 1e-6):
        V = pd.read_csv(path)
        _, _, z, p, s = _columns(V, None, depth=depth, vp=vp, vs=vs, need_xy=False)
        o = np.argsort(z, kind='stable')
        z, p, s = z[o], p[o], s[o]
        # layer tops -> an explicit staircase: each top becomes (top - w, value above), (top + w, value below),
        # with w = smooth_km/2 (w = eps gives the exact discontinuity)
        w = max(float(smooth_km) / 2.0, eps)
        zz, pp, ss = [z[0] - 1e3], [p[0]], [s[0]]
        for k in range(len(z)):
            if k:
                zz.append(z[k] - w); pp.append(p[k - 1]); ss.append(s[k - 1])
            zz.append(z[k] + (0.0 if k == 0 else w)); pp.append(p[k]); ss.append(s[k])
        zz.append(z[-1] + 1e3); pp.append(p[-1]); ss.append(s[-1])
        self.z = np.asarray(zz, float); self.v = {0: np.asarray(pp, float), 1: np.asarray(ss, float)}
        self.description = (f"layers1d:{path} ({len(z)} layers, constant within each, "
                            + (f"{len(z) - 1} discontinuit{'y' if len(z) == 2 else 'ies'}"
                               if smooth_km <= 0 else f"interfaces ramped over {smooth_km:g} km") + ")")

    def __call__(self, phase, X, Y, Z):
        col = np.interp(np.asarray(Z, dtype=float), self.z, self.v[phase])
        return np.ascontiguousarray(np.broadcast_to(col[None, None, :], (len(X), len(Y), len(Z))).copy())


class Profile1D:
    """Laterally uniform (1-D) model from a CSV of depth, Vp and Vs (or Vp/Vs); linear in depth between nodes.

    Column names and units are handled exactly as in `LayeredGridVelocity.from_csv`. Depths outside the file are
    held constant (the shallowest/deepest node).

    **Rows are nodes of a continuous profile, not layer tops.** Two rows (3 km, 5.9) and (8 km, 6.5) mean a
    gradient from 5.9 to 6.5 between 3 and 8 km, NOT a constant 5.9 km/s layer. To express a layer cake, repeat
    the depth: ..., (3, 4.2), (3, 5.9), (8, 5.9), (8, 6.5), ... Each repeat is a true discontinuity, which no grid
    can represent exactly -- see docs/velocity-models.md."""

    def __init__(self, path: str, depth=None, vp=None, vs=None):
        V = pd.read_csv(path)
        _, _, z, p, s = _columns(V, None, depth=depth, vp=vp, vs=vs, need_xy=False)
        o = np.argsort(z)
        self.z = z[o]; self.v = {0: p[o], 1: s[o]}
        steps = int(np.sum(np.diff(self.z) == 0))
        self.description = (f"profile1d:{path} ({len(self.z)} nodes, linear in depth between them"
                            + (f", {steps} discontinuit{'y' if steps == 1 else 'ies'}" if steps else
                               "; repeat a depth to make a true layer boundary") + ")")

    def __call__(self, phase, X, Y, Z):
        col = np.interp(np.asarray(Z, dtype=float), self.z, self.v[phase])
        return np.ascontiguousarray(np.broadcast_to(col[None, None, :], (len(X), len(Y), len(Z))).copy())


def _pick(cols, aliases, given=None):
    if given is not None:
        return given
    low = {c.lower().strip(): c for c in cols}
    for a in aliases:
        if a in low:
            return low[a]
    return None


def _columns(V, proj, lon=None, lat=None, depth=None, vp=None, vs=None, x=None, y=None, need_xy=True):
    """Map an arbitrary velocity table to (x_km, y_km, depth_km, vp_kms, vs_kms), detecting units.

    Detected: velocities in m/s (any value > 100) -> km/s; depths in m (any |value| > 1000) -> km; a column
    named elevation/elev/height, or depths that are mostly negative at the top, -> flipped to depth-positive-down;
    Vs from a Vp/Vs column when no Vs column exists.

    Every unit decision is PRINTED, because the heuristics can misfire (e.g. a shallow model with all depths
    < 1000 m given in metres is read as km) -- check the line and pass explicit column names if it is wrong."""
    notes = []
    zc = _pick(V.columns, ['depth_km', 'depth', 'z_km', 'z', 'dep'], depth)
    ec = _pick(V.columns, ['elevation', 'elev', 'height', 'altitude'])
    if zc is None and ec is None:
        raise ValueError(f'no depth/elevation column in {list(V.columns)}; pass depth=...')
    z = V[zc if zc is not None else ec].to_numpy(float)
    if np.nanmax(np.abs(z)) > 1000.0:              # metres
        z = z / 1000.0
        notes.append(f"'{zc or ec}' read as METRES (|z| > 1000 seen)")
    else:
        notes.append(f"'{zc or ec}' read as km")
    if zc is None:                                  # elevation -> depth
        z = -z
        notes.append(f"'{ec}' treated as elevation, flipped to depth")
    pc = _pick(V.columns, ['vp', 'vel_p', 'v_p', 'p_velocity', 'velp', 'vpkms'], vp)
    if pc is None:
        raise ValueError(f'no Vp column in {list(V.columns)}; pass vp=...')
    p = V[pc].to_numpy(float)
    sc = _pick(V.columns, ['vs', 'vel_s', 'v_s', 's_velocity', 'vels'], vs)
    if sc is not None:
        s = V[sc].to_numpy(float)
    else:
        rc = _pick(V.columns, ['vpvs', 'vp_vs', 'vpvs_ratio', 'ratio'])
        if rc is None:
            raise ValueError(f'no Vs or Vp/Vs column in {list(V.columns)}; pass vs=...')
        s = p / V[rc].to_numpy(float)
        notes.append(f"Vs derived from '{pc}'/'{rc}'")
    if np.nanmax(p) > 100.0:                        # m/s
        p, s = p / 1000.0, s / 1000.0
        notes.append(f"velocities read as M/S (values > 100 seen), converted to km/s")
    else:
        notes.append("velocities read as km/s")
    print(f"[gridtt] velocity file interpretation: {'; '.join(notes)}", flush=True)
    if not need_xy:
        return None, None, z, p, s
    xc = _pick(V.columns, ['x_km', 'x'], x); yc = _pick(V.columns, ['y_km', 'y'], y)
    if xc is not None and yc is not None:
        return V[xc].to_numpy(float), V[yc].to_numpy(float), z, p, s
    oc = _pick(V.columns, ['lon', 'longitude', 'long'], lon); ac = _pick(V.columns, ['lat', 'latitude'], lat)
    if oc is None or ac is None:
        raise ValueError(f'no lon/lat (or x/y) columns in {list(V.columns)}')
    if proj is None:
        raise ValueError('lon/lat model needs a projection (build_for_spider takes it from model.domain)')
    X, Y = proj(V[oc].to_numpy(float), V[ac].to_numpy(float))
    return np.asarray(X, float), np.asarray(Y, float), z, p, s


def open_velocity(path: str, proj=None, fmt: str = 'auto', smooth_km: float = 0.0):
    """Open any supported velocity file: `fmt` in {auto, ascii8, csv, profile1d}.

    auto: whitespace-separated with no header -> `ascii8`; a CSV with lon/lat (or x/y) -> a 3-D
    `LayeredGridVelocity`; a CSV with only depth and velocities -> a 1-D `Profile1D`.

    `profile1d` and `layers1d` read the SAME file as different media: `profile1d` interpolates between rows,
    `layers1d` treats each row as the top of a constant-velocity layer. Choose deliberately -- `auto` cannot tell
    them apart and picks `profile1d`."""
    fmt = (fmt or 'auto').lower()
    if fmt == 'ascii8':
        return LayeredGridVelocity.from_ascii8(path, proj)
    if fmt == 'profile1d':
        return Profile1D(path)
    if fmt in ('layers1d', 'layercake', 'nlloc'):
        return LayerCake1D(path, smooth_km=smooth_km)
    if fmt == 'csv':
        return LayeredGridVelocity.from_csv(path, proj)
    head = open(path).readline()
    if ',' not in head:
        return LayeredGridVelocity.from_ascii8(path, proj)
    cols = [c.strip().lower() for c in head.split(',')]
    if any(c in cols for c in ('lon', 'longitude', 'long', 'lat', 'latitude')) or ('x' in cols and 'y' in cols):
        return LayeredGridVelocity.from_csv(path, proj)
    return Profile1D(path)


class FunctionVelocity:
    """Analytic velocity, ``f(phase, gx, gy, gz) -> km/s`` on meshgrid arrays. Used by the tests."""

    def __init__(self, f, description: str = "function"):
        self.f = f; self.description = description

    def __call__(self, phase, X, Y, Z):
        gx, gy, gz = np.meshgrid(X, Y, Z, indexing='ij')
        return np.ascontiguousarray(np.broadcast_to(self.f(phase, gx, gy, gz), gx.shape).astype(np.float64))
