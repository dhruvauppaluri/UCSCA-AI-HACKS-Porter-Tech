import numpy as np

from so101_sim.camera import Frame
from so101_sim.vision import ColorObjectDetector, TableCalibration


def test_four_point_calibration():
    pixels = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    table = np.array([[0.2, -0.1], [0.3, -0.1], [0.3, 0.0], [0.2, 0.0]], dtype=np.float32)
    calibration = TableCalibration.from_points(pixels, table)
    assert np.allclose(calibration.pixel_to_xy(50, 50), [0.25, -0.05], atol=1e-5)


def test_detector_filters_yellow_robot_outside_pickup_zone():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[145:160, 145:160] = [230, 190, 30]
    rgb[110:125, 145:160] = [230, 190, 30]
    detections = ColorObjectDetector(TableCalibration.simulated()).detect(Frame(rgb, 1.0, 0))
    assert len(detections) == 1
    assert detections[0].name == "cylinder"
    assert detections[0].table_xy[1] < -0.025
