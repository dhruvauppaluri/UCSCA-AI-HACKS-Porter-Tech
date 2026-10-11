#!/usr/bin/env python3
"""Take a dated frame from the existing paper-camera MJPEG stream.

The stream server already owns /dev/video2. This program reads its HTTP feed,
so it does not compete for the camera device or either robot serial port.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2


CAMERAS = {
    "cam0": ("http://127.0.0.1:8767/cam0", "usb-icSpring_icspring_camera-video-index0"),
    "cam2": ("http://127.0.0.1:8765/cam2", "usb-icSpring_icspring_camera_202404160005-video-index0"),
}
DEFAULT_OUTPUT_DIR = Path.home() / "drawing_capture" / "data" / "camera"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Short name for the capture, such as blank or dot_1")
    parser.add_argument("--camera", choices=CAMERAS, default="cam0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.label):
        parser.error("label must contain only letters, digits, hyphens, or underscores")

    stream, camera_id = CAMERAS[args.camera]
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera stream: {stream}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stream opened but returned no frame")
    finally:
        cap.release()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    name = f"{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}_{args.label}"
    image_path = args.output_dir / f"{name}.jpg"
    metadata_path = args.output_dir / f"{name}.json"
    if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Failed to write {image_path}")
    metadata = {
        "label": args.label,
        "captured_at_utc": timestamp.isoformat(),
        "camera": camera_id,
        "stream": stream,
        "image": str(image_path),
        "width_px": frame.shape[1],
        "height_px": frame.shape[0],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
