# Local changes (zross)

Vendored 2026-09-22 from /oak-data/qhig/hawaii_magmatic_web/gridtt_send/gridtt (qhig's gridtt-0.1.0).
Installed editable into the `torch` env. Changes on top of the original:

1. `operator.py` — out-of-box source clamping is now *reported*: a `[gridtt] WARNING` is printed on the
   first clamped lookup and again each time the running total grows 10x, instead of only incrementing a
   counter nothing read. `strict=True` behavior unchanged.
2. `operator.py` — `table_index` no longer runs `torch.unique` (an O(N log N) sort of every batch);
   receivers are matched to stations by chunked direct `cdist` argmin against a float32 station copy
   (`sta_f` buffer). Same 0.05 km tolerance and failure behavior.
3. `velocity.py` — every unit/convention decision in `_columns` (m vs km, m/s vs km/s, elevation flip,
   Vs from Vp/Vs) is now printed as `[gridtt] velocity file interpretation: ...` so a misdetected file
   fails loud.
4. `tests/test_gridtt.py` — new: constant-velocity exactness of T and autograd gradients (factored FMM is
   exact there), unknown-receiver rejection, clamp counting + strict mode, Profile1D vs LayerCake1D media
   distinction, unit detection. `python -m pytest tests/` (6 tests).

5. NEW `radial2d` mode for 1-D velocity models (2026-09-22): `build_tables_1d` solves one axisymmetric
   (r, z) half-plane per station-phase with the same factored 2-D FMM (exact in a laterally uniform medium:
   rays stay in the vertical source-receiver plane), and `GridTTRadial` reads it back with Catmull-Rom
   bicubic, REFLECTING the stencil across the axis (T is even in r, so reflection is exact, keeps C1, and
   gives dT/dr -> 0 on the axis). 2-3 orders of magnitude smaller than 3-D tables at finer spacing.
   `load_gridtt` dispatches on `meta['mode']`; `gridtt build-spider --mode auto` picks radial for
   profile1d/layers1d models. `operator.py` was restructured around a shared `_TableModel` base
   (station lookup + out-of-box accounting).
6. Fix in the (2) lookup optimization: float32 `torch.cdist` defaults to the matmul identity, which loses
   ~0.1 km at ~300 km coordinate magnitudes and broke the 0.05 km receiver match (found on the Parkfield
   dataset). Now forced to `compute_mode='donot_use_mm_for_euclid_dist'`; regression test added.
7. `build_tables` / `build_tables_1d` now reject duplicate station names (they silently corrupted the
   name->table lut before; hyposvi/SPIDER dedupe stations, so builds must too).

8. `gridtt build` (new; `build-spider` kept as alias) accepts BOTH config layouts, auto-detected:
   SPIDER (nested io/model.domain) and hyposvi (flat station_file/lat_min/z_max...). The hyposvi route
   replicates hyposvi's own station conventions (network.station dedupe, x_km/y_km passthrough, elev_km -> z,
   station-min frame default), sizes the event box from catalog_infile when available (station extent + warning
   otherwise), and the derived `<config>_gridtt.json` drops the EikoNet architecture keys. For BOTH layouts all
   output paths in the derived config are suffixed `_gridtt` (we clobbered a finished EikoNet run's paths once);
   if the input config's directory is read-only, the derived config is written into the table directory instead.

9. Performance regression fix (2026-09-25): change (2) matched every row against every station
   (O(rows x stations)); at ~1500 stations that was ~98% of the step (4.45 s of 4.53 s per 3M-row
   batch). `table_index` now deduplicates receiver rows first (`torch.unique(dim=0)`) and runs the
   direct-path cdist on the <= n_sta unique receivers only: 0.006 s lookup, 0.039 s full fwd+bwd per
   batch (~116x). Co-located stations (identical xyz, hence identical tables) tie-break arbitrarily,
   exactly as before.

SPIDER integration: the `spider_patch/eikonet_loader_gridtt.patch` equivalent is applied natively in
~/git/SPIDER `spider/core/eikonet_loader.py` (gridtt branch before the eikonet import), so
`spider locate-map SPIDER_gridtt.json --device N` works without the spider_hook shim.

10. Co-located stations (2026-10-04): `table_index` now matches a receiver to the nearest station *that has a
    table for the requested phase*, within the same 0.05 km tolerance, instead of the nearest station regardless
    of phase. SCSN has pairs at identical coordinates (CI.WAS / CI.WAS2); when a model holds tables for only some
    of them or some phases (SPIDER-tomo solves per-minibatch subsets), the old lookup could land on the twin
    without the table and raise "no table for a requested phase". Receivers are now deduplicated per (xyz, phase).
    The distance error is unchanged; the phase error is raised only when no station within tolerance has the
    table. Test: `test_colocated_stations_resolve_by_phase`.

11. Fused lookup for samplers (2026-10-08, for ~/git/smcloc): `table_ids(sta_names, phases)` resolves table indices
    by station name once; `GridTT.lookup` / `GridTTRadial.lookup(t, src, grad=False)` read back T (and, with grad,
    dT/dsrc from the analytic derivative of the Catmull-Rom weights on the same stencil reads) for any leading shape,
    under no_grad, and return an `out` mask instead of counting clamps. Same interpolant as `forward`; `forward` is
    unchanged. New `catmull_rom_dweights`. Test: `test_fused_lookup_matches_forward_and_autograd` (both table types,
    out-of-table points, radial axis).
