"""Fixed, previously unseen cartoon dog strokes in the inset paper frame.

Coordinates are centimeters from the inset bottom-left. These paths are an
input to the existing controller, not fitted to any simulated output.
"""

from __future__ import annotations

import math


def ellipse(cx, cy, rx, ry, count=25):
    return [(cx + rx * math.cos(2 * math.pi * i / (count - 1)),
             cy + ry * math.sin(2 * math.pi * i / (count - 1)))
            for i in range(count)]


def dog_strokes():
    """Front-facing sitting dog, fixed within x=3.0..14.3, y=5.6..19.0 cm."""
    return [
        # Head and floppy ears.
        [(5.5, 15.8), (6.0, 17.5), (7.2, 18.5), (9.8, 18.5),
         (11.0, 17.5), (11.5, 15.8), (11.0, 14.0), (10.0, 12.9),
         (7.0, 12.9), (6.0, 14.0), (5.5, 15.8)],
        [(5.7, 17.3), (4.1, 18.7), (3.3, 18.3), (3.8, 14.6), (5.7, 15.1)],
        [(11.3, 17.3), (12.9, 18.7), (13.7, 18.3), (13.2, 14.6), (11.3, 15.1)],
        # Eyes, nose, muzzle, mouth.
        ellipse(7.3, 16.1, .25, .32, 13),
        ellipse(9.7, 16.1, .25, .32, 13),
        [(7.8, 14.6), (8.5, 14.2), (9.2, 14.6), (7.8, 14.6)],
        [(8.5, 14.2), (8.5, 13.7), (7.9, 13.3), (7.3, 13.5)],
        [(8.5, 13.7), (9.1, 13.3), (9.7, 13.5)],
        # Collar and tag.
        [(6.4, 12.8), (7.2, 12.2), (9.8, 12.2), (10.6, 12.8)],
        ellipse(8.5, 11.9, .37, .42, 13),
        # Sitting torso and front legs.
        [(6.7, 12.1), (5.5, 11.2), (5.0, 9.5), (5.2, 7.5),
         (6.0, 6.0), (7.1, 5.7), (7.5, 6.0), (7.5, 8.7)],
        [(10.3, 12.1), (11.5, 11.2), (12.0, 9.5), (11.8, 7.5),
         (11.0, 6.0), (9.9, 5.7), (9.5, 6.0), (9.5, 8.7)],
        [(7.5, 8.7), (7.5, 6.0), (8.0, 5.6), (9.0, 5.6),
         (9.5, 6.0), (9.5, 8.7)],
        [(5.6, 6.4), (4.7, 5.9), (4.9, 5.6), (7.2, 5.6)],
        [(11.4, 6.4), (12.3, 5.9), (12.1, 5.6), (9.8, 5.6)],
        # Raised tail on the right.
        [(11.8, 9.0), (13.3, 9.5), (14.2, 10.7),
         (14.3, 12.0), (13.8, 12.7)],
    ]
