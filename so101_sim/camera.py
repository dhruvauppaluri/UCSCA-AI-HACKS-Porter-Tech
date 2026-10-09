"""Interchangeable synthetic and XIAO camera frame sources."""

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

import cv2
import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray
    captured_at: float
    sequence: int


class CameraSource(Protocol):
    def get_frame(self) -> Frame: ...
    def close(self) -> None: ...


@dataclass
class CameraEffects:
    jpeg_quality: int = 38
    brightness_range: float = 0.12
    blur_probability: float = 0.08
    drop_probability: float = 0.02
    delay_frames: int = 2
    delay_jitter_frames: int = 1
    perspective_jitter_pixels: float = 2.0


class MuJoCoCamera:
    def __init__(self, env, effects: CameraEffects | None = None, seed: int = 0):
        self.env = env
        self.effects = effects or CameraEffects()
        self.rng = np.random.default_rng(seed)
        self.sequence = 0
        self.buffer: deque[Frame] = deque(maxlen=max(2, self.effects.delay_frames + self.effects.delay_jitter_frames + 2))

    def get_frame(self) -> Frame:
        rgb = self.env.render()
        brightness = self.rng.uniform(1 - self.effects.brightness_range, 1 + self.effects.brightness_range)
        rgb = np.clip(rgb.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        if self.effects.perspective_jitter_pixels > 0:
            height, width = rgb.shape[:2]
            corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            jitter = self.rng.uniform(-self.effects.perspective_jitter_pixels,
                                      self.effects.perspective_jitter_pixels, size=(4, 2)).astype(np.float32)
            warp = cv2.getPerspectiveTransform(corners, corners + jitter)
            rgb = cv2.warpPerspective(rgb, warp, (width, height), borderMode=cv2.BORDER_REPLICATE)
        if self.rng.random() < self.effects.blur_probability:
            rgb = cv2.GaussianBlur(rgb, (3, 3), 0.8)
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, self.effects.jpeg_quality])
        if not success:
            raise RuntimeError("JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        frame = Frame(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), monotonic(), self.sequence)
        self.sequence += 1
        if self.rng.random() >= self.effects.drop_probability or not self.buffer:
            self.buffer.append(frame)
        delay = max(0, self.effects.delay_frames + self.rng.integers(
            -self.effects.delay_jitter_frames, self.effects.delay_jitter_frames + 1))
        index = max(0, len(self.buffer) - 1 - delay)
        return self.buffer[index]

    def close(self):
        self.buffer.clear()


class XIAOCamera:
    """Read Seeed's HTTP MJPEG stream or any OpenCV-compatible camera URL."""

    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot open camera stream: {url}")
        self.sequence = 0

    def get_frame(self) -> Frame:
        ok, bgr = self.cap.read()
        if not ok:
            raise ConnectionError("XIAO camera frame unavailable")
        frame = Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), monotonic(), self.sequence)
        self.sequence += 1
        return frame

    def close(self):
        self.cap.release()
