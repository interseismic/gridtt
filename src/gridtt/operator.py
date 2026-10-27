"""GridTT: differentiable travel-time operators backed by precomputed eikonal tables.

Drop-in for SPIDER's EikoNet: ``forward(x[N, 7]) -> T[N, 1]`` with x = (src_x, src_y, src_z, rec_x, rec_y, rec_z,
phase) in SPIDER's projected km frame (Lambert azimuthal equal-area at the domain's lon_min/lat_min; z = depth, km).

Two operators share the format and the interface:

* ``GridTT``       -- one 3-D table per station and phase (any velocity model).
* ``GridTTRadial`` -- one 2-D (r, z) table per station and phase, for 1-D (laterally uniform) models. In a 1-D
  medium rays stay in the vertical source-receiver plane, so the 3-D field is exactly T(r_horizontal, z); the
  tables are 2-3 orders of magnitude smaller and can be solved at much finer spacing for the same budget.

Read-back is Catmull-Rom (tri/bi)cubic interpolation: interpolating (passes through the solved values) and C1
(value and first derivative continuous across cell faces), so autograd returns an exact, chatter-free dT/dx for
first-order optimisers (Adam, SGLD). Its support is 4 nodes per axis, so it cannot create structure at scales
longer than one cell. Trilinear interpolation is NOT a substitute: its gradient jumps at every cell face."""
from __future__ import annotations
import json, warnings
import numpy as np, torch

_OFF = (-1, 0, 1, 2)


def catmull_rom_weights(f: torch.Tensor) -> torch.Tensor:
    """f: [..., 1] fractional position in [0, 1) -> weights [..., 4] for nodes -1, 0, +1, +2."""
    return torch.cat([-.5 * f**3 + f**2 - .5 * f, 1.5 * f**3 - 2.5 * f**2 + 1,
                      -1.5 * f**3 + 2 * f**2 + .5 * f, .5 * f**3 - .5 * f**2], dim=-1)


def catmull_rom_dweights(f: torch.Tensor) -> torch.Tensor:
    """d/df of catmull_rom_weights: f [..., 1] -> [..., 4]."""
    return torch.cat([-1.5 * f**2 + 2 * f - .5, 4.5 * f**2 - 5 * f,
                      -4.5 * f**2 + 4 * f + .5, 1.5 * f**2 - f], dim=-1)


class _TableModel(torch.nn.Module):
    """Shared machinery: the flat table buffer, station lookup, and out-of-box accounting."""

    def __init__(self, tables: np.ndarray, meta: dict, dtype=torch.float32, strict: bool = False):
        super().__init__()
        self.h = float(meta['h']); self.hz = float(meta.get('hz', meta['h'])); self.n_sta = int(meta['n_sta'])
        self.strict = strict; self.n_clamped = 0; self._next_clamp_log = 1
        self.register_buffer('G', torch.from_numpy(np.ascontiguousarray(tables)).to(dtype))
        self.register_buffer('sta', torch.tensor(meta['sta_xyz'], dtype=torch.float64))
        # float32 copy for the nearest-station lookup: coords are O(1e3) km, so fp32 error is ~1e-4 km,
        # far below both station spacing and the 0.05 km match tolerance
        self.register_buffer('sta_f', self.sta.to(torch.float32))
        tb = meta['tables']
        phases = meta.get('phases', [0, 1]); lut = -np.ones((self.n_sta, max(phases) + 1), dtype=np.int64)
        for k, m in enumerate(tb): lut[meta['sta_names'].index(m['sta']) if 'sta_names' in meta else k % self.n_sta, m['phase']] = k
        self.register_buffer('lut', torch.from_numpy(lut))
        self.meta = {k: v for k, v in meta.items() if k not in ('tables', 'sta_xyz')}

    def table_index(self, rec: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Match receivers to stations by nearest neighbour, among the stations that have a table for the
        requested phase.

        Receiver rows are deduplicated first (a batch holds millions of rows but at most n_sta distinct receiver
        coordinates per phase), so the distance matrix is (n_unique x n_sta) instead of (n_rows x n_sta): matching
        every row against every station costs O(rows * stations) and dominated the whole step at ~1500 stations.
        Nearest-neighbour (rather than exact hashing) because SPIDER passes float32 km. compute_mode is forced to
        the direct path: the matmul form of cdist loses ~0.1 km at 300 km coordinate magnitudes in float32.

        Co-located stations are common (CI.WAS / CI.WAS2 share coordinates exactly), and a model may hold tables
        for only some of them or only some phases (SPIDER-tomo solves tables per minibatch). The nearest station
        regardless of phase could therefore be one without the requested table while an equally near one has it;
        the match is restricted to stations holding that phase's table. "No table" is raised only when no station
        within the 0.05 km tolerance has one; a receiver with no station at all within tolerance raises the
        distance error as before."""
        r = rec.detach().to(self.sta_f.dtype)
        ph = phase.reshape(-1).long()
        key = torch.cat([r, ph.to(r.dtype).unsqueeze(1)], dim=1)
        uq, inv = torch.unique(key, dim=0, return_inverse=True)
        d = torch.cdist(uq[:, :3], self.sta_f, compute_mode='donot_use_mm_for_euclid_dist')
        worst = float(d.min(dim=1).values.max()) if d.numel() else 0.0
        if worst > 0.05:
            raise ValueError(f'GridTT: a receiver is {worst:.3f} km from the nearest station table')
        ph_u = uq[:, 3].long()
        avail = (self.lut[:, ph_u] >= 0).T                       # [n_unique, n_sta]: station has this phase's table
        near, ju = torch.where(avail, d, torch.full_like(d, float('inf'))).min(dim=1)
        if bool((near > 0.05).any()):
            raise ValueError('GridTT: no table for a requested phase (no station within 0.05 km of the receiver has one)')
        return self.lut[ju, ph_u][inv]

    def table_ids(self, sta_names, phases) -> torch.Tensor:
        """Table index for each (station name, phase) pair, by name (no coordinate matching). Use with `lookup`
        when the station-phase of every datum is fixed, e.g. the picks of an event: resolve once, read many
        times. Raises if a station is unknown or has no table for the phase."""
        if 'sta_names' not in self.meta:
            raise ValueError('GridTT: tables were built without sta_names; table_ids needs them')
        pos = {n: i for i, n in enumerate(self.meta['sta_names'])}
        names = list(sta_names); ph = np.asarray(phases, dtype=np.int64).reshape(-1)
        missing = sorted({n for n in names if n not in pos})
        if missing:
            raise ValueError(f'GridTT: no tables for stations {missing[:10]}' + (' ...' if len(missing) > 10 else ''))
        lut = self.lut.cpu().numpy()
        ids = lut[np.array([pos[n] for n in names], dtype=np.int64), ph]
        if (ids < 0).any():
            raise ValueError('GridTT: no table for a requested (station, phase)')
        return torch.from_numpy(ids).to(self.lut.device)

    def _count_out_of_box(self, out_box: torch.Tensor) -> None:
        n_out = int(out_box.sum())
        if n_out:
            self.n_clamped += n_out
            if self.strict: raise ValueError(f'GridTT: {n_out} sources outside their table (need 1 cell of padding)')
            if self.n_clamped >= self._next_clamp_log:
                print(f'[gridtt] WARNING: {self.n_clamped} source lookups outside their table so far '
                      f'(edge-clamped: T flattens there, so the data gradient vanishes for those events; '
                      f'rebuild with a larger --margin/--z-max if this grows)', flush=True)
                self._next_clamp_log = max(self.n_clamped * 10, 1000)


class GridTT(_TableModel):
    """3-D tables: one (nx, ny, nz) volume per station and phase."""

    def __init__(self, tables: np.ndarray, meta: dict, dtype=torch.float32, strict: bool = False):
        super().__init__(tables, meta, dtype=dtype, strict=strict)
        tb = meta['tables']
        self.register_buffer('geom', torch.tensor([[m['off'], m['nx'], m['ny'], m['nz']] for m in tb], dtype=torch.int64))
        self.register_buffer('orig', torch.tensor([[m['x0'], m['y0'], m['z0']] for m in tb], dtype=torch.float64))
        self.register_buffer('hvec', torch.tensor([self.h, self.h, self.hz], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src, rec, ph = x[:, 0:3], x[:, 3:6], x[:, 6]
        t = self.table_index(rec, ph)
        off, nx, ny, nz = self.geom[t, 0], self.geom[t, 1], self.geom[t, 2], self.geom[t, 3]
        c = (src - self.orig[t].to(src.dtype)) / self.hvec.to(src.dtype)
        b = torch.floor(c); f = (c - b).unsqueeze(-1); b = b.long()
        if not torch.jit.is_tracing():
            nmax = torch.stack([nx, ny, nz], 1)
            self._count_out_of_box(((b < 1) | (b > nmax - 3)).any(1))
        w = catmull_rom_weights(f)                                    # [N, 3, 4]
        out = torch.zeros(x.shape[0], dtype=self.G.dtype, device=x.device)
        zero = torch.zeros_like(nx)
        for a_, oa in enumerate(_OFF):
            ii = (b[:, 0] + oa).clamp(zero, nx - 1)
            for b_, ob in enumerate(_OFF):
                jj = (b[:, 1] + ob).clamp(zero, ny - 1); wij = w[:, 0, a_] * w[:, 1, b_]
                for k_, oc in enumerate(_OFF):
                    kk = (b[:, 2] + oc).clamp(zero, nz - 1)
                    out = out + (wij * w[:, 2, k_]).to(self.G.dtype) * self.G.index_select(0, off + (ii * ny + jj) * nz + kk)
        return out.unsqueeze(-1).to(x.dtype)

    @torch.no_grad()
    def lookup(self, t: torch.Tensor, src: torch.Tensor, grad: bool = False):
        """Fused read-back at precomputed table indices (see `table_ids`), without an autograd graph.

        t: long [...]; src: [..., 3] (same leading shape). Returns T [...], dT/dsrc [..., 3] (None unless grad) and
        out [...] (True where the stencil leaves the table; T is edge-clamped there). Same interpolant as forward;
        the gradient uses the analytic derivative of the Catmull-Rom weights on the same stencil reads, so it costs
        a fraction of autograd and stores nothing. Out-of-box lookups are reported, not counted."""
        shp = t.shape; t = t.reshape(-1); src = src.reshape(-1, 3).to(self.G.dtype)
        off, nx, ny, nz = self.geom[t, 0], self.geom[t, 1], self.geom[t, 2], self.geom[t, 3]
        c = (src - self.orig[t].to(src.dtype)) / self.hvec.to(src.dtype)
        b = torch.floor(c); f = (c - b).unsqueeze(-1); b = b.long()
        out = ((b < 1) | (b > torch.stack([nx, ny, nz], 1) - 3)).any(1)
        w = catmull_rom_weights(f)                                    # [N, 3, 4]
        dw = catmull_rom_dweights(f) if grad else None
        T = torch.zeros(t.shape[0], dtype=self.G.dtype, device=src.device)
        if grad: Tx, Ty, Tz = torch.zeros_like(T), torch.zeros_like(T), torch.zeros_like(T)
        zero = torch.zeros_like(nx)
        for a_, oa in enumerate(_OFF):
            ii = (b[:, 0] + oa).clamp(zero, nx - 1)
            for b_, ob in enumerate(_OFF):
                jj = (b[:, 1] + ob).clamp(zero, ny - 1)
                s = torch.zeros_like(T); sz = torch.zeros_like(T) if grad else None
                for k_, oc in enumerate(_OFF):
                    kk = (b[:, 2] + oc).clamp(zero, nz - 1)
                    g = self.G.index_select(0, off + (ii * ny + jj) * nz + kk)
                    s = s + w[:, 2, k_] * g
                    if grad: sz = sz + dw[:, 2, k_] * g
                T = T + w[:, 0, a_] * w[:, 1, b_] * s
                if grad:
                    Tx = Tx + dw[:, 0, a_] * w[:, 1, b_] * s
                    Ty = Ty + w[:, 0, a_] * dw[:, 1, b_] * s
                    Tz = Tz + w[:, 0, a_] * w[:, 1, b_] * sz
        dT = torch.stack([Tx / self.h, Ty / self.h, Tz / self.hz], -1).reshape(*shp, 3) if grad else None
        return T.reshape(shp), dT, out.reshape(shp)


class GridTTRadial(_TableModel):
    """2-D (r, z) tables for 1-D media: r = horizontal source-station distance, z = source depth.

    The travel-time field of a 1-D medium is even in r, so the interpolation stencil is REFLECTED across the
    axis (node -1 reads node +1): exact for an even function, keeps C1, and gives dT/dr -> 0 at r = 0 as the
    physics requires. h is the radial spacing dr; hz the depth spacing."""

    def __init__(self, tables: np.ndarray, meta: dict, dtype=torch.float32, strict: bool = False):
        super().__init__(tables, meta, dtype=dtype, strict=strict)
        tb = meta['tables']
        self.register_buffer('geom', torch.tensor([[m['off'], m['nr'], m['nz']] for m in tb], dtype=torch.int64))
        self.register_buffer('z0', torch.tensor([m['z0'] for m in tb], dtype=torch.float64))
        names = meta['sta_names']
        sxyz = np.asarray(meta['sta_xyz'], dtype=np.float64)
        self.register_buffer('tsta', torch.tensor(
            np.stack([sxyz[names.index(m['sta'])] for m in tb]), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src, rec, ph = x[:, 0:3], x[:, 3:6], x[:, 6]
        t = self.table_index(rec, ph)
        off, nr, nz = self.geom[t, 0], self.geom[t, 1], self.geom[t, 2]
        d = src[:, :2] - self.tsta[t, :2].to(src.dtype)
        r = torch.sqrt((d * d).sum(1) + 1e-12)
        cr = r / self.h
        cz = (src[:, 2] - self.z0[t].to(src.dtype)) / self.hz
        br = torch.floor(cr); bz = torch.floor(cz)
        fr = (cr - br).unsqueeze(-1); fz = (cz - bz).unsqueeze(-1)
        br = br.long(); bz = bz.long()
        if not torch.jit.is_tracing():
            # the low-r side is handled exactly by reflection; only the far edge and z edges clamp
            self._count_out_of_box((br > nr - 3) | (bz < 1) | (bz > nz - 3))
        wr = catmull_rom_weights(fr); wz = catmull_rom_weights(fz)   # [N, 4] each
        out = torch.zeros(x.shape[0], dtype=self.G.dtype, device=x.device)
        for a_, oa in enumerate(_OFF):
            ii = torch.minimum((br + oa).abs(), nr - 1)              # reflect across the axis
            for k_, oc in enumerate(_OFF):
                kk = (bz + oc).clamp(torch.zeros_like(nz), nz - 1)
                out = out + (wr[:, a_] * wz[:, k_]).to(self.G.dtype) * self.G.index_select(0, off + ii * nz + kk)
        return out.unsqueeze(-1).to(x.dtype)

    @torch.no_grad()
    def lookup(self, t: torch.Tensor, src: torch.Tensor, grad: bool = False):
        """Fused read-back at precomputed table indices; same contract as GridTT.lookup. The radial derivative
        is mapped to (x, y) by (dx, dy) / r; it vanishes on the axis by the reflected stencil."""
        shp = t.shape; t = t.reshape(-1); src = src.reshape(-1, 3).to(self.G.dtype)
        off, nr, nz = self.geom[t, 0], self.geom[t, 1], self.geom[t, 2]
        d = src[:, :2] - self.tsta[t, :2].to(src.dtype)
        r = torch.sqrt((d * d).sum(1) + 1e-12)
        cr = r / self.h
        cz = (src[:, 2] - self.z0[t].to(src.dtype)) / self.hz
        br = torch.floor(cr); bz = torch.floor(cz)
        fr = (cr - br).unsqueeze(-1); fz = (cz - bz).unsqueeze(-1)
        br = br.long(); bz = bz.long()
        out = (br > nr - 3) | (bz < 1) | (bz > nz - 3)
        wr = catmull_rom_weights(fr); wz = catmull_rom_weights(fz)
        if grad: dwr = catmull_rom_dweights(fr); dwz = catmull_rom_dweights(fz)
        T = torch.zeros(t.shape[0], dtype=self.G.dtype, device=src.device)
        if grad: Tr, Tz = torch.zeros_like(T), torch.zeros_like(T)
        zero = torch.zeros_like(nz)
        for a_, oa in enumerate(_OFF):
            ii = torch.minimum((br + oa).abs(), nr - 1)
            s = torch.zeros_like(T); sz = torch.zeros_like(T) if grad else None
            for k_, oc in enumerate(_OFF):
                kk = (bz + oc).clamp(zero, nz - 1)
                g = self.G.index_select(0, off + ii * nz + kk)
                s = s + wz[:, k_] * g
                if grad: sz = sz + dwz[:, k_] * g
            T = T + wr[:, a_] * s
            if grad:
                Tr = Tr + dwr[:, a_] * s
                Tz = Tz + wr[:, a_] * sz
        if grad:
            Tr = Tr / (self.h * r)
            dT = torch.stack([Tr * d[:, 0], Tr * d[:, 1], Tz / self.hz], -1).reshape(*shp, 3)
        else:
            dT = None
        return T.reshape(shp), dT, out.reshape(shp)


def load_gridtt(path: str, device='cpu', dtype=torch.float32, strict: bool = False) -> torch.nn.Module:
    meta = json.load(open(f'{path}/meta.json'))
    tab = np.fromfile(f'{path}/tables.f32', dtype=np.float32, count=meta['n_cells'])
    if dtype == torch.float16 or dtype == torch.bfloat16:
        warnings.warn('GridTT: half-precision storage costs ~0.3 ms (fp16) / ~2.5 ms (bf16) median travel-time error '
                      'and much more in dT/dz at fine dz; keep float32 unless memory forces it.')
    cls = GridTTRadial if str(meta.get('mode', 'grid3d')) == 'radial2d' else GridTT
    return cls(tab, meta, dtype=dtype, strict=strict).to(device)
