"""General configuration of paths needed throughout the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..").resolve()
BLD = SRC.joinpath("..", "bld").resolve()
DATA_DIR = ROOT / "data"

BLD_inputs = BLD / "inputs"
BLD_results = BLD / "results"
BLD_figures = BLD / "figures"
BLD_data = BLD / "data"
BLD_final = BLD / "final"


__all__ = [
    "BLD",
    "ROOT",
    "SRC",
    "DATA_DIR",
    "BLD_inputs",
    "BLD_results",
    "BLD_figures",
    "BLD_data",
    "BLD_final",
]
