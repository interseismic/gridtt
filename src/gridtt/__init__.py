"""GridTT -- differentiable travel-time tables (factored 2nd-order FMM + Catmull-Rom read-back) for SPIDER."""
from .operator import GridTT, GridTTRadial, load_gridtt, catmull_rom_weights, catmull_rom_dweights
from .build import build_tables, build_tables_1d, station_lattice, solve_table
from .velocity import LayeredGridVelocity, FunctionVelocity, Profile1D, LayerCake1D, open_velocity
from .validate import validate_tables
from .spider import build_for_config, build_for_spider
from . import spider_hook
__version__ = '0.1.0'
