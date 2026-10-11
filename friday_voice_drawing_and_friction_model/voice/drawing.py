"""
drawing.py — Turns "7", "42" or "smiley" into pen paths for the Friday arm.

Every shape is built from PIECEWISE PARAMETRIC FUNCTIONS:
  - line(t) = P0 + t * (P1 - P0)                         for t in [0, 1]
  - arc(t)  = (cx + rx*cos(theta(t)), cy + ry*sin(theta(t)))
              with theta(t) = a0 + t * (a1 - a0)          for t in [0, 1]
A "stroke" is a chain of these pieces drawn with the pen DOWN.
Between strokes the pen lifts.

Each character is designed in its own unit box: x in [0,1], y in [0,1].
We then build a coordinate plane from the drawing-area boundaries and
scale/shift each character's box onto it.

Run on its own to preview without the arm:
    python3 drawing.py 42
    python3 drawing.py smiley
    python3 drawing.py all
(needs: pip3 install matplotlib  -- only for the preview picture)
"""

import json
import math
import sys

# ------------------------------------------------------------------ pieces ---
def line(x0, y0, x1, y1):
    return ("line", x0, y0, x1, y1)


def arc(cx, cy, rx, ry, a0_deg, a1_deg):
    """Ellipse arc from angle a0 to a1 (degrees, counter-clockwise positive)."""
    return ("arc", cx, cy, rx, ry, a0_deg, a1_deg)


def arc_point(cx, cy, rx, ry, a_deg):
    a = math.radians(a_deg)
    return cx + rx * math.cos(a), cy + ry * math.sin(a)


# ------------------------------------------------------- character designs ---
# Each entry: list of strokes; each stroke: list of pieces (line/arc) in order.
_5_arc_start = arc_point(0.5, 0.32, 0.3, 0.26, 135)

SHAPES = {
    "0": [[arc(0.5, 0.5, 0.35, 0.45, 90, 450)]],
    "1": [
        [line(0.3, 0.75, 0.5, 0.95), line(0.5, 0.95, 0.5, 0.05)],
        [line(0.3, 0.05, 0.7, 0.05)],
    ],
    "2": [[
        arc(0.5, 0.7, 0.3, 0.25, 160, -40),
        line(*arc_point(0.5, 0.7, 0.3, 0.25, -40), 0.2, 0.05),
        line(0.2, 0.05, 0.8, 0.05),
    ]],
    "3": [[
        arc(0.5, 0.73, 0.28, 0.22, 150, -90),
        arc(0.5, 0.28, 0.30, 0.23, 90, -150),
    ]],
    "4": [[
        line(0.65, 0.05, 0.65, 0.95),
        line(0.65, 0.95, 0.15, 0.35),
        line(0.15, 0.35, 0.85, 0.35),
    ]],
    "5": [[
        line(0.78, 0.95, 0.28, 0.95),
        line(0.28, 0.95, *_5_arc_start),
        arc(0.5, 0.32, 0.3, 0.26, 135, -140),
    ]],
    "6": [[
        arc(0.6, 0.3, 0.4, 0.62, 100, 180),   # the stem curving down
        arc(0.5, 0.3, 0.3, 0.27, 180, 540),   # the bottom loop
    ]],
    "7": [[line(0.15, 0.95, 0.85, 0.95), line(0.85, 0.95, 0.4, 0.05)]],
    "8": [
        [arc(0.5, 0.73, 0.25, 0.22, -90, 270)],
        [arc(0.5, 0.28, 0.30, 0.26, 90, 450)],
    ],
    "9": [[
        arc(0.5, 0.7, 0.3, 0.27, 0, 360),     # the top loop
        arc(0.4, 0.7, 0.4, 0.62, 0, -80),     # the tail curving down
    ]],
    "smiley": [
        [arc(0.5, 0.5, 0.45, 0.45, 0, 360)],              # face
        [arc(0.35, 0.64, 0.05, 0.06, 0, 360)],            # left eye
        [arc(0.65, 0.64, 0.05, 0.06, 0, 360)],            # right eye
        [arc(0.5, 0.5, 0.27, 0.27, 205, 335)],            # smile
    ],
}

CHAR_WIDTH = {"smiley": 1.0}   # box width relative to height; digits default below
DIGIT_WIDTH = 0.6
GAP = 0.15                     # space between characters, relative to height
MAX_CHARS = 4


# --------------------------------------------------------------- sampling ---
def sample_piece(piece, steps_per_unit=60):
    """Evaluate one piecewise function at evenly spaced t values."""
    if piece[0] == "line":
        _, x0, y0, x1, y1 = piece
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) * steps_per_unit) + 1)
        return [(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)) for i in range(n)]
    _, cx, cy, rx, ry, a0, a1 = piece
    approx_len = abs(math.radians(a1 - a0)) * (rx + ry) / 2
    n = max(8, int(approx_len * steps_per_unit) + 1)
    return [arc_point(cx, cy, rx, ry, a0 + (a1 - a0) * i / (n - 1)) for i in range(n)]


def sample_stroke(stroke):
    pts = []
    for piece in stroke:
        p = sample_piece(piece)
        pts.extend(p if not pts else p[1:])   # don't repeat the joining point
    return pts


# ------------------------------------------------------- coordinate plane ---
def tokens_for(content):
    """'42' -> ['4','2'];  'smiley' -> ['smiley']."""
    c = str(content).strip().lower().replace(" ", "")
    if c in ("smiley", "smileyface", "smile", "face"):
        return ["smiley"]
    if c.isdigit() and 1 <= len(c) <= MAX_CHARS:
        return list(c)
    raise ValueError(f"Can only draw 1-{MAX_CHARS} digits or 'smiley', got {content!r}")


def plan(content, area):
    """
    Build the coordinate plane from the area boundaries and lay the
    characters onto it. Returns a list of strokes in AREA coordinates
    (same units as the boundaries, e.g. cm), each a list of (x, y).

    area = {"x_min":..., "x_max":..., "y_min":..., "y_max":..., "margin":0.1}
    """
    toks = tokens_for(content)
    widths = [CHAR_WIDTH.get(t, DIGIT_WIDTH) for t in toks]
    total_w = sum(widths) + GAP * (len(toks) - 1)          # in "height units"

    aw = area["x_max"] - area["x_min"]
    ah = area["y_max"] - area["y_min"]
    usable = 1 - 2 * area.get("margin", 0.1)
    h = min(ah * usable, aw * usable / total_w)            # character height
    x = area["x_min"] + (aw - total_w * h) / 2             # center horizontally
    y = area["y_min"] + (ah - h) / 2                       # center vertically

    strokes = []
    for tok, w in zip(toks, widths):
        for stroke in SHAPES[tok]:
            strokes.append([(x + u * w * h, y + v * h) for u, v in sample_stroke(stroke)])
        x += (w + GAP) * h
    return strokes


class Page:
    """
    Writes like a typewriter on a coordinate plane built from the page boundaries.

    The cursor starts at the TOP-LEFT. Each character is drawn at the cursor,
    then the cursor moves RIGHT. If the next character won't fit before the right
    boundary, the cursor moves DOWN one line and back to the LEFT.
    When there is no room for another line, the page is full.

    area = {"x_min", "x_max", "y_min", "y_max"} in the arm's units (e.g. cm)
    char_height = height of one character, same units
    """

    def __init__(self, area, char_height, line_gap=0.4, margin=0.0):
        self.area = area
        self.h = char_height
        self.line_step = char_height * (1 + line_gap)
        self.margin = margin
        self.new_page()

    def new_page(self):
        self.x = self.area["x_min"] + self.margin
        self.y_top = self.area["y_max"] - self.margin          # top of the current line
        self.drawn = []                                         # all strokes on this page

    def _fits_on_line(self, w):
        return self.x + w * self.h <= self.area["x_max"] - self.margin + 1e-9

    def _line_fits_on_page(self):
        return self.y_top - self.h >= self.area["y_min"] + self.margin - 1e-9

    def place(self, content):
        """Return the strokes for `content` at the cursor and advance the cursor.
        Raises PageFull if it runs out of room (nothing is drawn in that case)."""
        toks = tokens_for(content)
        # Dry run on a copy of the cursor so we never draw half a number.
        x, y_top, out = self.x, self.y_top, []
        widths = [CHAR_WIDTH.get(t, DIGIT_WIDTH) for t in toks]
        total = (sum(widths) + GAP * (len(toks) - 1)) * self.h
        line_width = self.area["x_max"] - self.area["x_min"] - 2 * self.margin
        if x + total > self.area["x_max"] - self.margin + 1e-9 and total <= line_width:
            # Whole number won't fit on this line but fits on a fresh one: wrap first,
            # so "26" is never split as "2" / "6".
            x = self.area["x_min"] + self.margin
            y_top -= self.line_step
        for tok in toks:
            w = CHAR_WIDTH.get(tok, DIGIT_WIDTH)
            if w * self.h > self.area["x_max"] - self.area["x_min"] - 2 * self.margin:
                raise ValueError("Characters are too big for the page width")
            if x + w * self.h > self.area["x_max"] - self.margin + 1e-9:   # wrap
                x = self.area["x_min"] + self.margin
                y_top -= self.line_step
            if y_top - self.h < self.area["y_min"] + self.margin - 1e-9:
                raise PageFull()
            y0 = y_top - self.h
            for stroke in SHAPES[tok]:
                out.append([(x + u * w * self.h, y0 + v * self.h) for u, v in sample_stroke(stroke)])
            x += (w + GAP) * self.h
        self.x, self.y_top = x, y_top
        self.drawn.extend(out)
        return out


class PageFull(Exception):
    pass


def path_length(strokes):
    return sum(math.dist(a, b) for s in strokes for a, b in zip(s, s[1:]))


# ---------------------------------------------------------------- preview ---
def save_preview(strokes, area, filename, already_drawn=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6 * (area["y_max"] - area["y_min"]) / (area["x_max"] - area["x_min"])))
    ax.add_patch(plt.Rectangle((area["x_min"], area["y_min"]),
                               area["x_max"] - area["x_min"], area["y_max"] - area["y_min"],
                               fill=False, ls="--", color="gray"))
    for s in already_drawn:    # earlier drawings on the same page, in gray
        xs, ys = zip(*s)
        ax.plot(xs, ys, color="#bbbbbb", lw=2)
    prev = None
    for s in strokes:
        if prev is not None:   # pen-up travel
            ax.plot([prev[0], s[0][0]], [prev[1], s[0][1]], color="lightgray", lw=0.8, ls=":")
        xs, ys = zip(*s)
        ax.plot(xs, ys, color="black", lw=2)
        ax.plot(*s[0], "o", color="green", ms=4)
        prev = s[-1]
    ax.set_aspect("equal")
    ax.set_title("black = drawing now, gray = already on page, dotted = pen up, green = stroke start",
                 fontsize=8)
    fig.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "page":
    # Simulate several voice commands on one page:  python3 drawing.py page 42 7 smiley 2026 ...
    area = {"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 15}
    page = Page(area, char_height=3, margin=0.5)
    for i, cmd in enumerate(sys.argv[2:], 1):
        before = list(page.drawn)
        try:
            s = page.place(cmd)
        except PageFull:
            print(f"{cmd}: PAGE FULL")
            break
        save_preview(s, area, f"step{i}_{cmd}.png", already_drawn=before)
        print(f"{cmd}: drawn, cursor now at x={page.x:.1f}, line top y={page.y_top:.1f}")
    sys.exit()

if __name__ == "__main__":
    demo_area = {"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 10, "margin": 0.1}
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    items = ["0123", "4567", "89", "smiley"] if what == "all" else [what]
    for item in items:
        s = plan(item, demo_area)
        print(f"{item}: {len(s)} strokes, {path_length(s):.1f} units of ink")
        save_preview(s, demo_area, f"preview_{item}.png")
        print(f"  saved preview_{item}.png")
    if what != "all":
        with open("path.json", "w") as f:
            json.dump({"content": what, "area": demo_area, "strokes": s}, f)
        print("  saved path.json")
