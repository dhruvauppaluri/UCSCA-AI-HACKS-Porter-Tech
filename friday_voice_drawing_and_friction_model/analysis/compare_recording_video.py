"""Exploratory, image-based comparison for the archived 25-second circle run.

This isolates ink added between two video frames, rectifies the sheet using
manually selected corners, and compares the right-hand new strokes to the
corresponding MuJoCo contact trace. It does not validate an autonomous writing
model: the paper placement in that trace was aligned using the same run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/recordings/arm_motion_20261010_214924_video.mp4"
TRACE = ROOT / "simulation/examples/arm_motion_214924_driven/trace.csv"
OUT = ROOT / "simulation/examples/arm_motion_214924_driven"

# Coordinates were picked on a 1920 x 1080 view of the archived video at 4 s.
# The 275 mm long edge runs horizontally in this camera view; the 220 mm short
# edge runs vertically. These are approximate visual corners, not calibration.
REFERENCE_SIZE = (1920, 1080)
PAPER_CORNERS = np.float32([[915, 400], [1215, 399], [1263, 487], [905, 498]])
ROI = (750, 320, 1370, 570)
MM_TO_PX = 2
LONG_MM, SHORT_MM = 275, 220


def video_frame(reader, second: float, fps: float) -> np.ndarray:
    rgb = reader.get_data(round(second * fps))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, REFERENCE_SIZE)


def register_late_frame(early: np.ndarray, late: np.ndarray):
    x0, y0, x1, y1 = ROI
    gray_early = cv2.cvtColor(early[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray_late = cv2.cvtColor(late[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.01)
    key_early, des_early = sift.detectAndCompute(gray_early, None)
    key_late, des_late = sift.detectAndCompute(gray_late, None)
    if des_early is None or des_late is None:
        raise ValueError("Could not find enough video features to register frames")
    matches = cv2.BFMatcher().knnMatch(des_early, des_late, k=2)
    good = [m for m, n in matches if m.distance < .8 * n.distance]
    if len(good) < 25:
        raise ValueError(f"Only {len(good)} frame matches; registration is unreliable")
    early_xy = np.float32([key_early[m.queryIdx].pt for m in good])
    late_xy = np.float32([key_late[m.trainIdx].pt for m in good])
    homography, inliers = cv2.findHomography(late_xy, early_xy, cv2.RANSAC, 3)
    if homography is None or int(inliers.sum()) < 25:
        raise ValueError("Frame homography had too few inliers")
    shift = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=float)
    world_h = shift @ homography @ np.linalg.inv(shift)
    registered = cv2.warpPerspective(late, world_h, REFERENCE_SIZE)
    return registered, len(good), int(inliers.sum())


def rectify(frame: np.ndarray) -> np.ndarray:
    destination = np.float32([[0, 0], [LONG_MM * MM_TO_PX - 1, 0],
                              [LONG_MM * MM_TO_PX - 1, SHORT_MM * MM_TO_PX - 1],
                              [0, SHORT_MM * MM_TO_PX - 1]])
    transform = cv2.getPerspectiveTransform(PAPER_CORNERS, destination)
    return cv2.warpPerspective(frame, transform,
                               (LONG_MM * MM_TO_PX, SHORT_MM * MM_TO_PX))


def skeleton(binary: np.ndarray) -> np.ndarray:
    current = binary.copy()
    result = np.zeros_like(current)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(current):
        eroded = cv2.erode(current, element)
        result = cv2.bitwise_or(result, cv2.subtract(current, cv2.dilate(eroded, element)))
        current = eroded
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(str(args.video), format="ffmpeg")
    try:
        metadata = reader.get_meta_data()
        fps = metadata["fps"]
        if metadata["duration"] < 24:
            raise ValueError("Video is too short for the selected comparison frames")
        early = video_frame(reader, 4, fps)
        late = video_frame(reader, 24, fps)
    finally:
        reader.close()
    registered, matches, inliers = register_late_frame(early, late)
    before, after = rectify(early), rectify(registered)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Grayscale difference identifies darker pixels newly present by 24 s.
    # Limit the metric to the isolated right-hand strokes; changes elsewhere
    # include pen occlusion, old ink, and perspective-registration artifacts.
    change = before_gray.astype(np.int16) - after_gray.astype(np.int16)
    new_ink = ((change > 22) & (before_gray > 80) & (after_gray < 230)).astype(np.uint8)
    new_ink[:40] = 0
    new_ink[425:] = 0
    new_ink[:, :310] = 0
    new_ink[:, 445:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(new_ink, 8)
    filtered = np.zeros_like(new_ink)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= 8:
            filtered[labels == component] = 255
    actual_centerline = skeleton(filtered)

    with args.trace.open(newline="") as handle:
        trace = list(csv.DictReader(handle))
    predicted = []
    for row in trace:
        if row["ink"] != "1" or not 12 <= float(row["time_s"]) < 20:
            continue
        paper_x = float(row["ink_x_m"]) * 1000
        paper_y = float(row["ink_y_m"]) * 1000
        # The 275 mm paper axis appears horizontal in the video; the 220 mm
        # axis appears vertical. The inset is provisionally centered.
        u = round(MM_TO_PX * (paper_y + 20))
        v = round(MM_TO_PX * (paper_x + 25))
        if 310 <= u < 445 and 40 <= v < 425:
            predicted.append((u, v))
    predicted = np.asarray(predicted, dtype=int)
    if not len(predicted) or not cv2.countNonZero(actual_centerline):
        raise ValueError("No matching right-hand ink strokes were detected")
    predicted_mask = np.zeros_like(actual_centerline)
    predicted_mask[predicted[:, 1], predicted[:, 0]] = 255
    to_actual_mm = cv2.distanceTransform(255 - actual_centerline,
                                         cv2.DIST_L2, 5) / MM_TO_PX
    pred_distances = to_actual_mm[predicted[:, 1], predicted[:, 0]]
    to_prediction_mm = cv2.distanceTransform(255 - predicted_mask,
                                             cv2.DIST_L2, 5) / MM_TO_PX
    actual_distances = to_prediction_mm[actual_centerline > 0]

    # The plot intentionally shows source marks and the two path series together.
    # It is a visual consistency check, not a held-out accuracy score.
    comparison = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
    comparison[filtered > 0] = [192, 48, 39]
    for u, v in predicted:
        cv2.circle(comparison, (int(u), int(v)), 2, (20, 48, 160), -1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    extent = [0, LONG_MM, SHORT_MM, 0]
    axes[0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), extent=extent)
    axes[0].set_title("Video at 4 s: earlier ink already present")
    axes[1].imshow(comparison, extent=extent)
    axes[1].set_title("24 s video: new ink (red), simulated contact (blue)")
    for ax in axes:
        ax.set_xlabel("Along sheet's 275 mm edge (mm)")
        ax.set_ylabel("Along sheet's 220 mm edge (mm)")
    fig.suptitle("Same recording · right-hand strokes compared at approximate paper scale")
    fig.savefig(args.output_dir / "video_vs_mujoco.png", dpi=160)
    plt.close(fig)

    video_label = (str(args.video.relative_to(ROOT)) if args.video.is_relative_to(ROOT)
                   else str(args.video))
    report = {
        "video": video_label, "video_duration_s": metadata["duration"],
        "comparison_frames_s": [4, 24], "simulated_interval_s": [12, 20],
        "feature_matches": matches, "registration_inliers": inliers,
        "predicted_contact_points_compared": len(predicted),
        "new_ink_centerline_pixels": int(cv2.countNonZero(actual_centerline)),
        "predicted_to_new_ink_centerline_mean_mm": float(pred_distances.mean()),
        "predicted_to_new_ink_centerline_median_mm": float(np.median(pred_distances)),
        "predicted_points_within_10_mm_fraction": float(np.mean(pred_distances <= 10)),
        "new_ink_centerline_to_predicted_mean_mm": float(actual_distances.mean()),
        "limitations": [
            "Video and CSV have matching run names and durations, but no hardware timecode.",
            "4 s frame already has ink; only newly visible right-hand strokes are scored.",
            "Perspective corners were picked visually and paper inset is assumed centered.",
            "Image differencing, blur, occlusion, and marker width affect the extracted centerline.",
            "MuJoCo paper placement was adjusted using this same recording; these are not held-out accuracy metrics.",
        ],
    }
    (args.output_dir / "video_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "registration_inliers", "predicted_contact_points_compared",
        "predicted_to_new_ink_centerline_mean_mm",
        "predicted_points_within_10_mm_fraction",
        "new_ink_centerline_to_predicted_mean_mm")}, indent=2))


if __name__ == "__main__":
    main()
