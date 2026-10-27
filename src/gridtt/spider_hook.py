"""Run SPIDER with GridTT **without modifying SPIDER**.

SPIDER's inputs are unchanged (same dtimes / station / events / config); the only thing that differs is which
object is built for the travel-time model. `spider.cli` binds `load_eikonet_model` at import time, so rebinding it
before the command runs is enough:

    python -m gridtt.spider_hook locate-map SPIDER.json --device 0

The GridTT table directory is taken from `model.model_file`; it is recognised either by
`model.eikonet.model_kind == "gridtt"` or simply by the directory containing `meta.json` + `tables.f32`.
Anything else falls through to the normal EikoNet loader, so one command works for both.
"""
from __future__ import annotations
import os, sys


def _is_table_dir(path) -> bool:
    try:
        return os.path.isdir(path) and os.path.exists(os.path.join(path, "meta.json")) \
            and os.path.exists(os.path.join(path, "tables.f32"))
    except Exception:
        return False


def install() -> None:
    """Rebind spider.cli.load_eikonet_model so GridTT table directories are loaded as the forward model."""
    import spider.cli as cli
    from .operator import load_gridtt
    original = cli.load_eikonet_model

    def _load(params, device):
        cfg = ((params.get("model") or {}).get("eikonet") or {}) if isinstance(params, dict) else {}
        mf = params.get("model_file") if isinstance(params, dict) else None
        if str(cfg.get("model_kind", "")).strip().lower() == "gridtt" or _is_table_dir(mf):
            print(f"[gridtt] travel-time operator: interpolated FMM tables at {mf}", flush=True)
            return load_gridtt(str(mf), device=device)
        return original(params=params, device=device)

    cli.load_eikonet_model = _load


def main(argv=None) -> int:
    install()
    from spider.cli import main as spider_main
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    return spider_main()


if __name__ == "__main__":
    sys.exit(main())
