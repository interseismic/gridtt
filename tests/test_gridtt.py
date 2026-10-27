"""Unit tests for the GridTT operator, velocity readers and lookup logic.

The strongest check is exactness on a constant-velocity medium: factored FMM is exact there
(tau1 = 1/v is the solution of the factored eikonal), so T = |x - x_sta| / v up to interpolation
error of a smooth function -- any indexing, orientation or weighting bug shows up immediately.
"""
import numpy as np
import pytest
import torch

from gridtt import build_tables, load_gridtt
from gridtt.velocity import FunctionVelocity, LayerCake1D, Profile1D

VP, VS = 6.0, 3.46
STA = np.array([[3.7, -2.3, 1.1]])  # deliberately off-origin to catch axis transposition


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("tab"))
    vel = FunctionVelocity(lambda ph, x, y, z: np.where(ph == 0, VP, VS) * np.ones_like(x),
                           description="const test")
    build_tables(STA, (-8.0, 8.0, -8.0, 8.0), vel, out, dx=0.5, dz=0.25,
                 z_top=-1.0, z_max=10.0, margin=2.0, names=["TEST"], log=lambda *a: None)
    return out


def _points(n, seed=0, rmin=1.5):
    rng = np.random.default_rng(seed)
    src = rng.uniform([-7, -7, 0.0], [7, 7, 9.0], size=(n, 3))
    r = np.linalg.norm(src - STA[0], axis=1)
    return src[r > rmin]


def test_constant_velocity_exact(tables):
    model = load_gridtt(tables)
    src = _points(4000)
    for ph, v in ((0, VP), (1, VS)):
        x = np.column_stack([src, np.tile(STA[0], (len(src), 1)), np.full(len(src), ph)])
        T = model(torch.tensor(x, dtype=torch.float32)).squeeze(1).detach().numpy()
        r = np.linalg.norm(src - STA[0], axis=1)
        err = np.abs(T - r / v)
        assert err.max() < 1e-3, f"phase {ph}: max |T - r/v| = {err.max():.2e} s"


def test_constant_velocity_gradient(tables):
    model = load_gridtt(tables)
    # r > 2.5 km (5 cells): the exact gradient's curvature ~1/r means the cubic's derivative
    # error concentrates near the source; validate.py excludes r < 2 cells for the same reason
    src = _points(2000, seed=1, rmin=2.5)
    x = torch.tensor(np.column_stack([src, np.tile(STA[0], (len(src), 1)), np.zeros(len(src))]),
                     dtype=torch.float32, requires_grad=True)
    T = model(x)
    g, = torch.autograd.grad(T.sum(), x)
    r = np.linalg.norm(src - STA[0], axis=1, keepdims=True)
    g_exact = (src - STA[0]) / (r * VP)
    err = np.abs(g[:, :3].numpy() - g_exact)
    assert err.max() < 2e-3, f"max gradient error {err.max():.2e} s/km (|g| = {1/VP:.3f})"


def test_receiver_lookup_rejects_unknown_station(tables):
    model = load_gridtt(tables)
    x = torch.zeros((1, 7)); x[0, 3:6] = torch.tensor([50.0, 50.0, 0.0])
    with pytest.raises(ValueError, match="from the nearest station"):
        model(x)


def test_out_of_box_source_counted_and_strict_raises(tables, capsys):
    model = load_gridtt(tables)
    x = torch.zeros((1, 7))
    x[0, 0:3] = torch.tensor([200.0, 0.0, 5.0])  # far outside the table
    x[0, 3:6] = torch.tensor(STA[0], dtype=torch.float32)
    model(x)
    assert model.n_clamped == 1
    assert "outside their table" in capsys.readouterr().out
    strict = load_gridtt(tables, strict=True)
    with pytest.raises(ValueError, match="outside their table"):
        strict(x)


def test_profile_vs_layercake_are_different_media(tmp_path):
    f = tmp_path / "vel.csv"
    f.write_text("depth,vp,vs\n0.0,4.0,2.3\n10.0,6.0,3.5\n")
    prof, cake = Profile1D(str(f)), LayerCake1D(str(f))
    z = np.array([5.0])
    vp_prof = prof(0, [0.0], [0.0], z)[0, 0, 0]   # linear: 5.0 km/s at mid-depth
    vp_cake = cake(0, [0.0], [0.0], z)[0, 0, 0]   # layer top: constant 4.0 to 10 km
    assert abs(vp_prof - 5.0) < 1e-9
    assert abs(vp_cake - 4.0) < 1e-9


@pytest.fixture(scope="module")
def radial_tables(tmp_path_factory):
    from gridtt import build_tables_1d
    out = str(tmp_path_factory.mktemp("rtab"))
    vel = FunctionVelocity(lambda ph, x, y, z: np.where(ph == 0, VP, VS) * np.ones_like(x),
                           description="const radial test")
    build_tables_1d(STA, (-8.0, 8.0, -8.0, 8.0), vel, out, dr=0.5, dz=0.25,
                    z_top=-1.0, z_max=10.0, margin=2.0, names=["TEST"], log=lambda *a: None)
    return out


def test_radial_constant_velocity_exact(radial_tables):
    from gridtt.operator import GridTTRadial
    model = load_gridtt(radial_tables)
    assert isinstance(model, GridTTRadial)
    src = _points(4000)
    for ph, v in ((0, VP), (1, VS)):
        x = np.column_stack([src, np.tile(STA[0], (len(src), 1)), np.full(len(src), ph)])
        T = model(torch.tensor(x, dtype=torch.float32)).squeeze(1).detach().numpy()
        r = np.linalg.norm(src - STA[0], axis=1)   # full 3-D distance
        err = np.abs(T - r / v)
        assert err.max() < 1e-3, f"phase {ph}: max |T - r/v| = {err.max():.2e} s"


def test_radial_gradient_and_axis(radial_tables):
    model = load_gridtt(radial_tables)
    src = _points(2000, seed=2, rmin=2.5)
    x = torch.tensor(np.column_stack([src, np.tile(STA[0], (len(src), 1)), np.zeros(len(src))]),
                     dtype=torch.float32, requires_grad=True)
    g, = torch.autograd.grad(model(x).sum(), x)
    r = np.linalg.norm(src - STA[0], axis=1, keepdims=True)
    err = np.abs(g[:, :3].numpy() - (src - STA[0]) / (r * VP))
    assert err.max() < 2e-3, f"max gradient error {err.max():.2e} s/km"

    # directly on the axis (source vertically below the station): finite gradient,
    # zero horizontal component, T = |dz| / v
    x0 = torch.tensor([[STA[0, 0], STA[0, 1], 6.1, *STA[0], 0.0]], dtype=torch.float32, requires_grad=True)
    T0 = model(x0)
    g0, = torch.autograd.grad(T0.sum(), x0)
    assert abs(float(T0) - (6.1 - STA[0, 2]) / VP) < 1e-3
    assert torch.isfinite(g0).all()
    assert abs(float(g0[0, 0])) < 1e-3 and abs(float(g0[0, 1])) < 1e-3
    assert abs(float(g0[0, 2]) - 1.0 / VP) < 2e-3


def test_radial_matches_grid3d_on_1d_model(tmp_path):
    """Same 1-D gradient medium, same spacing: the radial and 3-D builds must agree."""
    from gridtt import build_tables, build_tables_1d
    vel = FunctionVelocity(lambda ph, x, y, z: 4.0 + 0.15 * np.clip(z, 0, None) + 0.5 * ph,
                           description="gradient 1d")
    d3, dr_ = str(tmp_path / "g3"), str(tmp_path / "r2")
    build_tables(STA, (-8, 8, -8, 8), vel, d3, dx=0.5, dz=0.25, z_top=-1, z_max=10, margin=2,
                 names=["TEST"], log=lambda *a: None)
    build_tables_1d(STA, (-8, 8, -8, 8), vel, dr_, dr=0.5, dz=0.25, z_top=-1, z_max=10, margin=2,
                    names=["TEST"], log=lambda *a: None)
    A, B = load_gridtt(d3), load_gridtt(dr_)
    src = _points(3000, seed=3, rmin=2.0)
    x = torch.tensor(np.column_stack([src, np.tile(STA[0], (len(src), 1)), np.zeros(len(src))]),
                     dtype=torch.float32)
    dT = (A(x) - B(x)).abs().squeeze(1).detach().numpy()
    assert np.median(dT) < 2e-3 and dT.max() < 2e-2, f"radial vs 3-D: median {np.median(dT):.2e}, max {dT.max():.2e} s"


def test_station_lookup_precise_at_large_coordinates(tmp_path):
    """Regression: float32 matmul-form cdist loses ~0.1 km at ~300 km coordinates, enough to break
    the 0.05 km receiver-to-station match. The lookup must use the direct-distance path."""
    from gridtt import build_tables_1d
    sta = STA + np.array([[300.0, 340.0, 0.0]])
    vel = FunctionVelocity(lambda ph, x, y, z: VP * np.ones_like(x))
    out = str(tmp_path / "far")
    build_tables_1d(sta, (292.0, 312.0, 332.0, 352.0), vel, out, dr=1.0, dz=0.5,
                    z_top=-1.0, z_max=10.0, margin=2.0, names=["FAR"], phases=(0,), log=lambda *a: None)
    model = load_gridtt(out)
    x = torch.tensor(np.column_stack([sta + [[3.0, 4.0, 5.0]], sta, [[0.0]]]), dtype=torch.float32)
    T = model(x)  # must not raise the 0.05 km mismatch error
    assert abs(float(T) - np.sqrt(50.0) / VP) < 2e-3


def test_duplicate_station_names_rejected(tmp_path):
    from gridtt import build_tables_1d
    vel = FunctionVelocity(lambda ph, x, y, z: VP * np.ones_like(x))
    with pytest.raises(ValueError, match="duplicate station names"):
        build_tables_1d(np.vstack([STA, STA]), (-8, 8, -8, 8), vel, str(tmp_path / "d"),
                        names=["A", "A"], log=lambda *a: None)


def _write_common_inputs(tmp_path):
    import pandas as pd
    st = pd.DataFrame({"network": ["NW"] * 3, "station": ["A", "B", "A"],   # one duplicate row on purpose
                       "latitude": [32.02, 32.10, 32.02], "longitude": [130.02, 130.12, 130.02],
                       "elev_km": [0.2, 0.4, 0.2]})
    ev = pd.DataFrame({"latitude": [32.03, 32.08], "longitude": [130.04, 130.10], "depth": [4.0, 7.0]})
    vel = tmp_path / "vel.csv"
    vel.write_text("depth,vp,vs\n0.0,5.0,2.9\n10.0,6.5,3.7\n")
    stf, evf = tmp_path / "st.csv", tmp_path / "ev.csv"
    st.to_csv(stf, index=False); ev.to_csv(evf, index=False)
    return str(stf), str(evf), str(vel)


def test_build_for_config_spider_layout(tmp_path):
    import json
    from gridtt import build_for_config, load_gridtt
    stf, evf, vel = _write_common_inputs(tmp_path)
    cfg = {"io": {"station_file": stf, "catalog_infile": evf,
                  "catalog_outfile": str(tmp_path / "out.csv"), "samples_outfile": str(tmp_path / "s.h5"),
                  "checkpoint_dir": str(tmp_path / "ckpt") + "/"},
           "model": {"model_file": "x.pt", "domain": {"lon_min": 130.0, "lat_min": 32.0,
                                                      "z_min": -1.0, "z_max": 12.0}},
           "observability": {"wandb": {"run_name": "r1"}}}
    cfgf = tmp_path / "SPIDER.json"
    cfgf.write_text(json.dumps(cfg))
    M = build_for_config(str(cfgf), vel, str(tmp_path / "tab"), dx=1.0, dz=0.5, log=lambda *a: None)
    assert M["mode"] == "radial2d" and M["n_sta"] == 2      # deduped, radial auto-chosen
    load_gridtt(str(tmp_path / "tab"))
    out = json.loads((tmp_path / "SPIDER_gridtt.json").read_text())
    assert out["model"]["eikonet"]["model_kind"] == "gridtt"
    assert out["io"]["catalog_outfile"].endswith("out_gridtt.csv")
    assert out["io"]["checkpoint_dir"].endswith("ckpt_gridtt/")
    assert out["observability"]["wandb"]["run_name"] == "r1_gridtt"


def test_build_for_config_hyposvi_layout(tmp_path):
    import json
    from gridtt import build_for_config, load_gridtt
    stf, evf, vel = _write_common_inputs(tmp_path)
    cfg = {"station_file": stf, "catalog_infile": evf, "output_file": str(tmp_path / "origins.csv"),
           "lat_min": 32.0, "lon_min": 130.0, "z_min": -1.0, "z_max": 12.0,
           "n_hidden": 24, "n_blocks": 4, "likelihood": "huber"}
    cfgf = tmp_path / "hyposvi.json"
    cfgf.write_text(json.dumps(cfg))
    M = build_for_config(str(cfgf), vel, str(tmp_path / "tab"), dx=1.0, dz=0.5, log=lambda *a: None)
    assert M["mode"] == "radial2d" and M["lat0"] == 32.0
    load_gridtt(str(tmp_path / "tab"))
    out = json.loads((tmp_path / "hyposvi_gridtt.json").read_text())
    assert out["model_kind"] == "gridtt" and out["model_file"].endswith("tab")
    assert out["output_file"].endswith("origins_gridtt.csv")
    assert "n_hidden" not in out and "n_blocks" not in out    # EikoNet arch keys dropped
    assert out["likelihood"] == "huber"


def test_unit_detection_metres_and_ms(tmp_path, capsys):
    f = tmp_path / "vel_ms.csv"
    f.write_text("depth,vp,vs\n0,4000,2300\n5000,6000,3500\n")
    p = Profile1D(str(f))
    out = capsys.readouterr().out
    assert "METRES" in out and "M/S" in out
    assert abs(p(0, [0.0], [0.0], np.array([5.0]))[0, 0, 0] - 6.0) < 1e-9


def test_colocated_stations_resolve_by_phase(tmp_path):
    """Two stations at identical coordinates (CI.WAS / CI.WAS2 in SCSN). A lookup must pick, among the
    stations within tolerance, one that has a table for the requested phase, and fail only when none has.
    Regression for a SPIDER-tomo crash: tables for (A, P) and (A2, S) in one model, query (A2, S)."""
    import json
    from gridtt import build_tables_1d
    from gridtt.operator import GridTTRadial
    vel = FunctionVelocity(lambda ph, x, y, z: np.where(ph == 0, VP, VS) * np.ones_like(x))
    out = str(tmp_path / "co")
    build_tables_1d(np.vstack([STA, STA]), (-8, 8, -8, 8), vel, out, dr=0.5, dz=0.25, z_top=-1.0, z_max=10.0,
                    margin=2.0, names=["A", "A2"], log=lambda *a: None)
    meta = json.load(open(out + "/meta.json"))
    tab = np.fromfile(out + "/tables.f32", dtype=np.float32, count=meta["n_cells"])

    def subset(keep):
        tbs, parts, off = [], [], 0
        for m in meta["tables"]:
            if (m["sta"], m["phase"]) in keep:
                n = m["nr"] * m["nz"]
                parts.append(tab[m["off"]:m["off"] + n]); tbs.append(dict(m, off=off)); off += n
        return GridTTRadial(np.concatenate(parts), dict(meta, tables=tbs, n_cells=off))

    src = np.array([[3.0, 4.0, 5.0]])

    def q(model, phase, rec=STA):
        return float(model(torch.tensor(np.column_stack([src, rec, [[float(phase)]]]), dtype=torch.float32)))

    full = load_gridtt(out)
    T_p, T_s = q(full, 0), q(full, 1)
    dist = float(np.linalg.norm(src[0] - STA[0]))
    assert abs(T_p - dist / VP) < 2e-3 and abs(T_s - dist / VS) < 2e-3
    m = subset({("A", 0), ("A2", 1)})                     # the crash case
    assert abs(q(m, 0) - T_p) < 1e-6 and abs(q(m, 1) - T_s) < 1e-6
    m = subset({("A2", 1)})                               # only one table: right phase works, wrong phase says so
    assert abs(q(m, 1) - T_s) < 1e-6
    with pytest.raises(ValueError, match="no table for a requested phase"):
        q(m, 0)
    with pytest.raises(ValueError, match="km from the nearest station"):
        q(m, 1, rec=STA + [[1.0, 0.0, 0.0]])            # no station within tolerance: distance error


@pytest.mark.parametrize("which", ["tables", "radial_tables"])
def test_fused_lookup_matches_forward_and_autograd(which, request):
    """lookup(table_ids, src, grad=True) is the same interpolant as forward, and its analytic gradient equals
    autograd's, including points that leave the table (edge-clamped) and, for radial tables, the axis."""
    model = load_gridtt(request.getfixturevalue(which))
    rng = np.random.default_rng(5)
    src = rng.uniform([-9, -9, -1.5], [9, 9, 11.0], size=(3000, 3))     # some outside the table on purpose
    src = np.vstack([src, [[STA[0, 0], STA[0, 1], 4.0]]])               # on the axis
    for ph in (0, 1):
        x = torch.tensor(np.column_stack([src, np.tile(STA[0], (len(src), 1)), np.full(len(src), ph)]),
                         dtype=torch.float32, requires_grad=True)
        T_ref = model(x).squeeze(1)
        g_ref, = torch.autograd.grad(T_ref.sum(), x)
        t = model.table_ids(["TEST"] * len(src), [ph] * len(src))
        T, dT, out = model.lookup(t, torch.tensor(src, dtype=torch.float32), grad=True)
        assert torch.allclose(T, T_ref.detach(), atol=1e-5)
        assert torch.allclose(dT, g_ref[:, :3], atol=1e-4)
        T2, dT2, out2 = model.lookup(t[:3000].reshape(50, 60), torch.tensor(src[:3000], dtype=torch.float32).reshape(50, 60, 3))
        assert T2.shape == (50, 60) and dT2 is None and torch.equal(T2.reshape(-1), T[:3000])
        assert out.any() and not out.all()
    with pytest.raises(ValueError):
        model.table_ids(["NOPE"], [0])
