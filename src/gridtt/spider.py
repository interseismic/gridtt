"""Build GridTT tables from a locator config: the same stations, events and frame the locator uses.

Accepts BOTH config layouts and detects which one it was given:

* SPIDER  -- nested: ``io.station_file`` / ``io.catalog_infile`` / ``model.domain`` (lon_min, lat_min, z_min, z_max)
* hyposvi -- flat:   ``station_file`` / ``catalog_infile`` (optional) / ``lat_min``/``lon_min`` / ``z_min``/``z_max``

Both routes write ``<config>_gridtt.json`` next to the input: the same config with ``model_file`` pointing at the
tables, ``model_kind: "gridtt"``, EikoNet architecture keys dropped (hyposvi), and every OUTPUT path suffixed
``_gridtt`` so running it never overwrites the EikoNet run's results."""
from __future__ import annotations
import json, os, copy
import numpy as np, pandas as pd
from pyproj import Proj
from .build import build_tables, build_tables_1d
from .velocity import LayerCake1D, Profile1D, open_velocity

_EIKONET_ARCH_KEYS = ('n_hidden', 'n_blocks', 'n_fourier', 'use_fourier', 'phase_emb_dim', 't2_log_scale')


def spider_projection(cfg: dict) -> Proj:
    d = cfg['model']['domain']
    return Proj(proj='laea', lat_0=float(d['lat_min']), lon_0=float(d['lon_min']), datum='WGS84', units='km')


def _suffixed(path, tag='_gridtt'):
    root, ext = os.path.splitext(str(path))
    return root + tag + ext


def _read_stations(path: str, proj, log):
    """Stations in projected km, replicating each locator's own conventions (dedupe on network.station;
    x_km/y_km/z_km passthrough; depth[km] or elev_km -> z; z = depth, negative above sea level)."""
    st = pd.read_csv(path)
    if {'network', 'station'}.issubset(st.columns):
        n0 = len(st)
        st = st.drop_duplicates(subset=['network', 'station']).reset_index(drop=True)
        if len(st) < n0:
            log(f'stations: dropped {n0 - len(st)} duplicate network.station rows (matching the locators)')
        names = (st.network.astype(str) + '.' + st.station.astype(str)).tolist()
    else:
        names = [f'STA{i}' for i in range(len(st))]
    if {'x_km', 'y_km'}.issubset(st.columns):
        sx, sy = st.x_km.to_numpy(float), st.y_km.to_numpy(float)
    else:
        if proj is None:
            raise ValueError('station file has lon/lat but the config gives no projection origin (lat_min/lon_min)')
        sx, sy = proj(st.longitude.to_numpy(float), st.latitude.to_numpy(float))
    if 'z_km' in st.columns:
        sz = st.z_km.to_numpy(float)
    elif 'elev_km' in st.columns:
        sz = -st.elev_km.to_numpy(float)
    elif 'depth' in st.columns:
        sz = st.depth.to_numpy(float)
    else:
        sz = np.zeros(len(st))
    return np.c_[np.asarray(sx, float), np.asarray(sy, float), sz], names, st


def _event_box(catalog_infile, proj, sta_xyz, log):
    """Map-view box of the event volume; falls back to the station extent when no usable catalog exists."""
    if catalog_infile and os.path.exists(str(catalog_infile)):
        ev = pd.read_csv(catalog_infile)
        cols = {c.lower(): c for c in ev.columns}
        if 'longitude' in cols and 'latitude' in cols and proj is not None:
            ex, ey = proj(ev[cols['longitude']].to_numpy(float), ev[cols['latitude']].to_numpy(float))
            return float(np.min(ex)), float(np.max(ex)), float(np.min(ey)), float(np.max(ey))
        if 'x(km)' in cols and 'y(km)' in cols:
            ex, ey = ev[cols['x(km)']].to_numpy(float), ev[cols['y(km)']].to_numpy(float)
            return float(ex.min()), float(ex.max()), float(ey.min()), float(ey.max())
        log(f'WARNING: {catalog_infile} has no usable longitude/latitude columns; using the station extent as '
            f'the event box')
    else:
        log('WARNING: no catalog_infile to size the event volume; using the station extent as the event box. '
            'If events can occur outside the network, pass an events file or raise --margin.')
    return (float(sta_xyz[:, 0].min()), float(sta_xyz[:, 0].max()),
            float(sta_xyz[:, 1].min()), float(sta_xyz[:, 1].max()))


def _build(cfg_kind, sta_xyz, names, box, vel, out_dir, dx, dz, z_top, z_max, margin, n_jobs, mode, extra, log):
    if mode not in ('auto', 'grid3d', 'radial2d'):
        raise ValueError(f"mode must be auto/grid3d/radial2d, got {mode!r}")
    is_1d = isinstance(vel, (Profile1D, LayerCake1D))
    if mode == 'radial2d' and not is_1d:
        raise ValueError('mode=radial2d requires a 1-D velocity model (profile1d/layers1d)')
    use_radial = is_1d if mode == 'auto' else (mode == 'radial2d')
    if mode == 'auto':
        log(f"table mode: {'radial2d (1-D model -> axisymmetric 2-D tables)' if use_radial else 'grid3d'} "
            f"(chosen automatically; override with --mode)")
    builder = build_tables_1d if use_radial else build_tables
    spacing = dict(dr=dx, dz=dz) if use_radial else dict(dx=dx, dz=dz)
    return builder(sta_xyz, box, vel, out_dir, z_top=z_top, z_max=z_max, margin=margin,
                   names=names, n_jobs=n_jobs, log=log, extra_meta=extra, **spacing)


def build_for_config(config_path: str, velocity_file: str, out_dir: str, velocity_format: str = 'auto',
                     dx: float = 0.5, dz: float = 0.2, z_max: float | None = None, margin: float = 6.0,
                     n_jobs: int = 1, write_config: bool = True, smooth_km: float = 0.0, mode: str = 'auto',
                     log=print) -> dict:
    """Build tables for either a SPIDER or a hyposvi config (layout auto-detected)."""
    cfg = json.load(open(config_path))
    is_spider = isinstance(cfg.get('io'), dict) and isinstance(cfg.get('model'), dict)
    log(f'config layout: {"SPIDER" if is_spider else "hyposvi"} ({config_path})')

    if is_spider:
        d = cfg['model']['domain']
        lat0, lon0 = float(d['lat_min']), float(d['lon_min'])
        station_file, catalog_infile = cfg['io']['station_file'], cfg['io'].get('catalog_infile')
        z_top = float(d.get('z_min', -5.0))
        z_max = float(z_max if z_max is not None else d['z_max'])
    else:
        lat0 = cfg.get('lat_min', cfg.get('lat0')); lon0 = cfg.get('lon_min', cfg.get('lon0'))
        station_file, catalog_infile = cfg['station_file'], cfg.get('catalog_infile')
        z_top = float(cfg.get('z_min', -5.0))
        if z_max is None and 'z_max' not in cfg:
            raise ValueError('z_max not in the config; pass --z-max')
        z_max = float(z_max if z_max is not None else cfg['z_max'])

    if lat0 is None or lon0 is None:
        st_ll = pd.read_csv(station_file)
        if {'latitude', 'longitude'}.issubset(st_ll.columns):
            lat0, lon0 = float(st_ll.latitude.min()), float(st_ll.longitude.min())
            log(f'no lat_min/lon_min in config; defaulting the frame to the station minimum '
                f'lat0={lat0}, lon0={lon0} (hyposvi convention) -- consider pinning it in the config')
    proj = (Proj(proj='laea', lat_0=float(lat0), lon_0=float(lon0), datum='WGS84', units='km')
            if lat0 is not None and lon0 is not None else None)

    sta_xyz, names, _ = _read_stations(station_file, proj, log)
    box = _event_box(catalog_infile, proj, sta_xyz, log)
    deep_note = f'table volume: x {box[0]:.1f}..{box[1]:.1f} y {box[2]:.1f}..{box[3]:.1f} z {z_top}..{z_max} km'
    log(deep_note)
    vel = open_velocity(velocity_file, proj, velocity_format, smooth_km=smooth_km)
    log(f'velocity model: {getattr(vel, "description", vel)}')

    extra = dict(source_config=os.path.abspath(config_path),
                 velocity_file=os.path.abspath(velocity_file), velocity_format=velocity_format,
                 dx=dx, dz=dz, margin=margin)
    if lat0 is not None and lon0 is not None:
        extra.update(lon0=float(lon0), lat0=float(lat0))
    M = _build('spider' if is_spider else 'hyposvi', sta_xyz, names, box, vel, out_dir,
               dx, dz, z_top, z_max, margin, n_jobs, mode, extra, log)

    if write_config:
        c2 = copy.deepcopy(cfg)
        if is_spider:
            c2['model']['model_file'] = os.path.abspath(out_dir)
            c2['model'].setdefault('eikonet', {})['model_kind'] = 'gridtt'
            io = c2['io']
            for key in ('catalog_outfile', 'samples_outfile', 'output_file'):
                if key in io: io[key] = _suffixed(io[key])
            if 'checkpoint_dir' in io:
                io['checkpoint_dir'] = str(io['checkpoint_dir']).rstrip('/') + '_gridtt/'
            wb = c2.get('observability', {}).get('wandb', {})
            if 'run_name' in wb: wb['run_name'] = str(wb['run_name']) + '_gridtt'
        else:
            c2['model_file'] = os.path.abspath(out_dir)
            c2['model_kind'] = 'gridtt'
            for key in _EIKONET_ARCH_KEYS: c2.pop(key, None)
            if lat0 is not None and lon0 is not None:
                c2.setdefault('lat_min', float(lat0)); c2.setdefault('lon_min', float(lon0))
            for key in ('output_file', 'catalog_outfile', 'ssst_outfile', 'svi_outfile',
                        'svi_particles_outfile', 'phase_rows_outfile'):
                if key in c2: c2[key] = _suffixed(c2[key])
        p2 = os.path.splitext(config_path)[0] + '_gridtt.json'
        try:
            json.dump(c2, open(p2, 'w'), indent=1)
        except OSError:
            # config lives in a read-only tree (e.g. a colleague's directory): keep the config with the tables
            p2 = os.path.join(out_dir, os.path.basename(p2))
            json.dump(c2, open(p2, 'w'), indent=1)
        log(f'{"SPIDER" if is_spider else "hyposvi"} config for GridTT -> {p2} (output paths suffixed _gridtt)')
    return M


# Backwards-compatible name (the README and older scripts call this).
def build_for_spider(*args, **kwargs) -> dict:
    return build_for_config(*args, **kwargs)
