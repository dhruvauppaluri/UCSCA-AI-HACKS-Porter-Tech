"""Create a tabletop homography from four measured correspondences."""

import argparse
from pathlib import Path

import numpy as np

from .vision import TableCalibration


def _pairs(values: list[str]) -> np.ndarray:
    return np.asarray([[float(x) for x in value.split(",")] for value in values], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Map four camera pixels to four tabletop coordinates in meters")
    parser.add_argument("--pixels", nargs=4, required=True, metavar="U,V")
    parser.add_argument("--table-xy", nargs=4, required=True, metavar="X,Y")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    calibration = TableCalibration.from_points(_pairs(args.pixels), _pairs(args.table_xy))
    calibration.save(args.out)
    print(f"Saved calibration to {args.out.resolve()}")


if __name__ == "__main__":
    main()
