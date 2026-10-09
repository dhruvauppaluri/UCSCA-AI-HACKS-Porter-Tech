"""Small calibrated detector for the three color-coded prototype objects.

The detector's label, confidence, pixel center, and table XY contract is shared
by simulated and real camera sources. At the event the HSV front end can be
replaced with an object detector without changing the robot controller.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .camera import Frame


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    pixel_center: tuple[float, float]
    table_xy: tuple[float, float]
    captured_at: float


class TableCalibration:
    def __init__(self, homography: np.ndarray):
        self.h = np.asarray(homography, dtype=float)
        if self.h.shape != (3, 3):
            raise ValueError("homography must be 3x3")

    @classmethod
    def from_points(cls, pixels: np.ndarray, table_xy: np.ndarray):
        pixels = np.asarray(pixels, dtype=np.float32)
        table_xy = np.asarray(table_xy, dtype=np.float32)
        if pixels.shape != (4, 2) or table_xy.shape != (4, 2):
            raise ValueError("four pixel and four tabletop XY points are required")
        return cls(cv2.getPerspectiveTransform(pixels, table_xy))

    @classmethod
    def simulated(cls, width: int = 320, height: int = 240, fovy_deg: float = 55.0,
                  camera_xyz=(0.25, 0.0, 0.75)):
        half_y = camera_xyz[2] * np.tan(np.deg2rad(fovy_deg / 2))
        meters_per_pixel = 2 * half_y / height
        # Camera image x points along world +x; image y down points world -y.
        return cls(np.array([[meters_per_pixel, 0, camera_xyz[0] - width / 2 * meters_per_pixel],
                             [0, -meters_per_pixel, camera_xyz[1] + height / 2 * meters_per_pixel],
                             [0, 0, 1]], dtype=float))

    def pixel_to_xy(self, x: float, y: float) -> tuple[float, float]:
        v = self.h @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"homography": self.h.tolist()}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path):
        return cls(np.asarray(json.loads(path.read_text())["homography"], dtype=float))


class ColorObjectDetector:
    def __init__(self, calibration: TableCalibration, min_area: int = 22,
                 pickup_bounds: tuple[float, float, float, float] = (0.14, 0.35, -0.19, -0.025)):
        self.calibration = calibration
        self.min_area = min_area
        self.pickup_bounds = pickup_bounds

    def detect(self, frame: Frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        masks = {
            "box": cv2.inRange(hsv, (0, 80, 55), (11, 255, 255)) | cv2.inRange(hsv, (170, 80, 55), (179, 255, 255)),
            "bottle": cv2.inRange(hsv, (36, 60, 45), (90, 255, 255)),
            "cylinder": cv2.inRange(hsv, (16, 65, 55), (37, 255, 255)),
        }
        found = []
        for name, mask in masks.items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    break
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                px, py = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
                xy = self.calibration.pixel_to_xy(px, py)
                xmin, xmax, ymin, ymax = self.pickup_bounds
                if not (xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax):
                    continue
                # Confidence here is a visibility heuristic, not a learned probability.
                confidence = float(min(1.0, area / 160.0))
                found.append(Detection(name, confidence, (px, py), xy, frame.captured_at))
                break
        return found
