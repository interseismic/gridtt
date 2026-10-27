import argparse
from .spider import build_for_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog='gridtt', description='Differentiable travel-time tables for SPIDER and hyposvi')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, hlp in (('build', 'build P and S tables for the stations/events/frame of a SPIDER or hyposvi config'),
                      ('build-spider', 'alias of build (kept for compatibility; also accepts hyposvi configs)')):
        b = sub.add_parser(name, help=hlp)
        b.add_argument('config'); b.add_argument('--velocity', required=True)
        b.add_argument('--velocity-format', choices=['auto', 'ascii8', 'csv', 'profile1d', 'layers1d'], default='auto')
        b.add_argument('--out', required=True); b.add_argument('--dx', type=float, default=0.5)
        b.add_argument('--dz', type=float, default=0.2); b.add_argument('--z-max', type=float, default=None)
        b.add_argument('--margin', type=float, default=6.0); b.add_argument('--jobs', type=int, default=1)
        b.add_argument('--smooth-interfaces', type=float, default=0.0,
                       help='layers1d only: ramp every layer boundary over this many km (try 2 x dz)')
        b.add_argument('--mode', choices=['auto', 'grid3d', 'radial2d'], default='auto',
                       help='auto: 1-D velocity models get axisymmetric 2-D (r, z) tables (orders of magnitude '
                            'smaller, dx becomes the radial spacing); 3-D models get full 3-D tables')
    v = sub.add_parser('validate', help='check tables against a finer reference, for your own model and region')
    v.add_argument('tables'); v.add_argument('--velocity', default=None)
    v.add_argument('--velocity-format', choices=['auto', 'ascii8', 'csv', 'profile1d', 'layers1d'], default='auto')
    v.add_argument('--refine', type=int, default=2); v.add_argument('--stations', type=int, default=6)
    v.add_argument('--points', type=int, default=200000); v.add_argument('--jobs', type=int, default=1)
    v.add_argument('--gate', type=float, default=0.01); v.add_argument('--json', default=None)
    v.add_argument('--keep-reference', action='store_true'); v.add_argument('--reference-dir', default=None)

    a = ap.parse_args(argv)
    if a.cmd in ('build', 'build-spider'):
        build_for_config(a.config, a.velocity, a.out, a.velocity_format, a.dx, a.dz, a.z_max, a.margin, a.jobs,
                         smooth_km=a.smooth_interfaces, mode=a.mode)
    elif a.cmd == 'validate':
        from .validate import validate_tables
        from .velocity import open_velocity
        vel = None
        if a.velocity:
            import json as _json
            from pyproj import Proj
            m = _json.load(open(f'{a.tables}/meta.json'))
            proj = (Proj(proj='laea', lon_0=float(m['lon0']), lat_0=float(m['lat0']), datum='WGS84', units='km')
                    if 'lon0' in m and 'lat0' in m else None)
            vel = open_velocity(a.velocity, proj, a.velocity_format)
        r = validate_tables(a.tables, velocity=vel, refine=a.refine, n_sta=a.stations, n_points=a.points,
                            n_jobs=a.jobs, gate_hf=a.gate, out_json=a.json, ref_dir=a.reference_dir,
                            keep_ref=a.keep_reference)
        return 0 if r['passed'] else 1


if __name__ == '__main__':
    main()
