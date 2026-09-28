#!/usr/bin/env python3
"""
Calibration helper for the Q-learning line follower (ev3dev2).
=============================================================
Run this FIRST, before training, and again on demo day or whenever
the lighting / mat changes.

It prints the color sensor's live reflected-light intensity (0-100).
Slowly sweep the sensor over:
  - the dark floor            -> LOW readings
  - the bright/white line     -> HIGH readings
  - the edge between them     -> the midpoint becomes TARGET_INTENSITY

Stop with Ctrl-C (over SSH) or the EV3 back/exit button.

Improvements over the old version:
  1. ROBUST STATS: floor/line levels are taken from the 5th and 95th
     percentiles of ALL samples, not the raw min/max, so a single
     flicker (shadow, lifting the robot, pointing at a table edge)
     can no longer poison the calibration.
  2. NO HAND-COPYING: results are written to calibration.json in the
     CURRENT directory. rl_line_follower.py reads that file at
     startup automatically, so there is nothing to edit by hand
     (this is exactly how the old TARGET=13 vs log=23 mismatch
     happened).
  3. SAFE LOGGING: the human-readable log is appended next to the
     script instead of a hardcoded /home/robot/RL_Grp2/ path, and a
     write failure can no longer crash the calibration.
"""

import json
import os
import time
from datetime import datetime

from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import ColorSensor

# Files are written to the directory this script lives in, so the main
# script (kept in the same folder) finds calibration.json automatically.
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON = os.path.join(HERE, "calibration.json")
LOG_FILE = os.path.join(HERE, "calibration_log.txt")

# Margin between the detected floor/line levels and the lost-line
# thresholds (same "a little above / below" idea as before).
THRESHOLD_MARGIN = 5

# Ignore the most extreme 5% of samples at each end.
PERCENTILE = 0.05

color_sensor = ColorSensor(INPUT_1)
color_sensor.mode = 'COL-REFLECT'


def percentile(sorted_vals, p):
    """Value at fraction p (0..1) of a sorted list, nearest-rank."""
    if not sorted_vals:
        return 0
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


print("Reflected light intensity (0-100).")
print("Sweep the sensor slowly over floor / line / edge for ~20-30 s.")
print("Press Ctrl-C (or the EV3 back button) to stop.\n")

samples = []
try:
    while True:
        v = color_sensor.reflected_light_intensity
        samples.append(v)
        s = sorted(samples)
        lo = percentile(s, PERCENTILE)          # robust floor level
        hi = percentile(s, 1.0 - PERCENTILE)    # robust line level
        print("now={:3d}   floor~={:3d}   line~={:3d}   target~={:3d}   (n={})".format(
            v, lo, hi, (lo + hi) // 2, len(samples)))
        time.sleep(0.15)
except KeyboardInterrupt:
    pass

if len(samples) < 30:
    print("\nOnly {} samples collected -- sweep for longer (aim for 100+).".format(
        len(samples)))
    print("Nothing was saved.")
    raise SystemExit(1)

s = sorted(samples)
floor_level = percentile(s, PERCENTILE)
line_level = percentile(s, 1.0 - PERCENTILE)
raw_min, raw_max = s[0], s[-1]

target = (floor_level + line_level) // 2
lost_low = min(floor_level + THRESHOLD_MARGIN, 100)
lost_high = max(line_level - THRESHOLD_MARGIN, 0)

# Sanity check: an edge follower needs a usable window between the
# two thresholds. If floor and line are too close, calibration is bad
# (sensor too high, wrong surface, or not enough sweeping).
if lost_high - lost_low < 10:
    print("\nWARNING: contrast window is only {} wide ({}..{}).".format(
        lost_high - lost_low, lost_low, lost_high))
    print("Check sensor height (5-10 mm above the mat) and sweep again.")

print("\n--- Calibration summary (robust, 5th/95th percentile) ---")
print("Samples            : {}".format(len(samples)))
print("Raw min/max        : {} / {}   (ignored if outliers)".format(raw_min, raw_max))
print("Floor level        : {}".format(floor_level))
print("Line level         : {}".format(line_level))
print("TARGET_INTENSITY   = {}".format(target))
print("LOST_LINE_LOW      = {}".format(lost_low))
print("LOST_LINE_HIGH     = {}".format(lost_high))

# --- Machine-readable handoff: rl_line_follower.py reads this file ---
calib = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "samples": len(samples),
    "floor_level": floor_level,
    "line_level": line_level,
    "TARGET_INTENSITY": target,
    "LOST_LINE_LOW": lost_low,
    "LOST_LINE_HIGH": lost_high,
}
try:
    with open(CALIB_JSON, "w") as f:
        json.dump(calib, f, indent=2)
    print("\nSaved -> {}".format(CALIB_JSON))
    print("rl_line_follower.py will pick these values up automatically.")
except Exception as e:
    print("\nWARNING: could not write {}: {}".format(CALIB_JSON, e))
    print("Copy the three values above into rl_line_follower.py by hand.")

# --- Human-readable log (append; failure here must never crash us) ---
try:
    with open(LOG_FILE, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Calibration Log - {}\n".format(calib["timestamp"]))
        f.write("=" * 50 + "\n")
        f.write("Samples           : {}\n".format(len(samples)))
        f.write("Raw min/max       : {} / {}\n".format(raw_min, raw_max))
        f.write("Floor (p5)        : {}\n".format(floor_level))
        f.write("Line  (p95)       : {}\n".format(line_level))
        f.write("TARGET_INTENSITY  = {}\n".format(target))
        f.write("LOST_LINE_LOW     = {}\n".format(lost_low))
        f.write("LOST_LINE_HIGH    = {}\n".format(lost_high))
        f.write("=" * 50 + "\n")
    print("Log appended -> {}".format(LOG_FILE))
except Exception as e:
    print("WARNING: could not write log file: {}".format(e))
