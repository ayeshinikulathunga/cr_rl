#!/usr/bin/env python3
"""
Q-learning line follower -- corrected edition (ev3dev2)
=======================================================
Standalone file. It does NOT modify or import rl_line_follower.py,
rl_line_follower-new.py or rl-new.py, and it never writes q_table.pkl,
so those scripts and their Q-table keep working exactly as before.
This file keeps its own Q-table in q_table_fixed.pkl. It only READS
calibration.json (written by calibrate.py) and, for `audit_legacy` /
`table_legacy`, q_table.pkl.

Fixes for the three logical errors found in q_table.pkl / rl-new.py:

  1. Only FWD-L was ever trained, so FWD-R and REV fell back to
     "straight" in every state and could not follow the line.
     -> Epsilon and episode counts are stored PER MODE. Previously one
        shared epsilon meant a second mode started training at the
        first mode's exhausted ~0.08 and barely explored.
     -> `seed_fr` initialises FWD-R from the (correctly) mirrored FWD-L
        values; `train_fr` then fine-tunes them.
     -> `run` uses a mode's own Q-values once it has
        MIN_TRAINED_EPISODES, falls back to mirrored FWD-L for FWD-R,
        and refuses to switch into an untrained REV instead of driving
        blind. `status` shows what is trained and what to do next.
     -> The line search is edge-aware. The old one always turned the
        FWD-L way on "dark" and stopped at the first edge it crossed,
        so a right-edge robot that lost the line came back on the LEFT
        edge, and FWD-R training silently learned the left edge.

  2. rl-new.py mirrored FWD-L -> FWD-R twice (flipped the bin AND the
     action). The two flips cancel, so the "right-edge" follower
     steered exactly like a left-edge follower.
     -> Right-edge following is left-edge following seen in a mirror at
        the SAME brightness: keep the bin, mirror only the action.

  3. The FWD-L policy was non-monotonic (dark-side bins 1-2 steered the
     same way as the bright side, "straight" was never chosen).
     -> Bins are laid out around the target: one centre bin on the
        target, N_SIDE bins on the dark side, N_SIDE on the bright side,
        each 2-3 intensity units wide. The old 8 equal bins were ~1.9
        wide (inside the sensor noise), and the target sat inside a bin
        instead of having its own. In simulation (8 fresh 45-episode
        trainings each) the new layout lost the line ~20% less often and
        never produced a direction flip; the old one did in 1 of 8.
     -> `audit` checks that each mode's greedy steering DIRECTION is
        monotonic across the bins (dark side one way, bright side the
        other) and that the two extreme bins steer opposite ways.
        `seed_fr` refuses to copy a FWD-L that fails.
     -> The Q-table records the calibration it was trained under and
        warns if calibration.json has changed since.
     Timing (STEP_SLEEP), sensing and reward are exactly as in
     rl_line_follower.py: a longer control period, median-filtered
     reads and reward shaping were all tried and did not help.

Execution order (each step depends only on the ones before it):
  1. python3 calibrate.py                                (existing, unchanged)
  2. python3 rl_line_follower_fixed.py audit_legacy      read-only, no motors
  3. python3 rl_line_follower_fixed.py test_motors
  4. python3 rl_line_follower_fixed.py train_fl  [episodes]  repeat until audit PASS
  5. python3 rl_line_follower_fixed.py seed_fr               needs step 4 PASS
     python3 rl_line_follower_fixed.py train_fr  [episodes]  fine-tune
  6. python3 rl_line_follower_fixed.py train_rev [episodes]  repeat until audit PASS
  7. python3 rl_line_follower_fixed.py audit
  8. python3 rl_line_follower_fixed.py run [seconds]
  `status` prints this checklist with what is done and what is next.
  With no arguments (e.g. launched from the brick's file browser) it
  runs DEFAULT_COMMAND.

During `run` the brick buttons work as in rl_line_follower.py:
  LEFT  -> forward, follow LEFT edge
  RIGHT -> forward, follow RIGHT edge
  DOWN  -> reverse (only once REV is trained)
  UP    -> back to the last forward mode

Written for the ev3dev Python 3.5 runtime (no f-strings).
"""

import json
import os
import pickle
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON = os.path.join(HERE, "calibration.json")
# Own Q-table (absolute path, so the working directory doesn't matter).
Q_TABLE_PATH = os.path.join(HERE, "q_table_fixed.pkl")
# The old scripts' table: read-only here.
LEGACY_Q_TABLE_PATH = os.path.join(HERE, "q_table.pkl")
LEGACY_N_BINS = 8

# Command used when the script is started without arguments.
DEFAULT_COMMAND = ["status"]

# =====================================================================
# 1. CALIBRATION (same defaults and override rules as rl_line_follower.py)
# =====================================================================
TARGET_INTENSITY = 21      # edge between white line and dark floor
LOST_LINE_LOW    = 9       # <= this  -> fully on dark floor (lost)
LOST_LINE_HIGH   = 34      # >= this  -> fully on white line (lost)
CALIB_TIMESTAMP  = None


def _load_calibration():
    global TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH, CALIB_TIMESTAMP
    try:
        with open(CALIB_JSON) as f:
            c = json.load(f)
        target = int(c["TARGET_INTENSITY"])
        low = int(c["LOST_LINE_LOW"])
        high = int(c["LOST_LINE_HIGH"])
    except IOError:
        print("No calibration.json found -> using built-in defaults "
              "(target={} low={} high={}). Run calibrate.py!".format(
                  TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH))
        return
    except (KeyError, ValueError, TypeError) as e:
        print("calibration.json is malformed ({}) -> using defaults.".format(e))
        return
    TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH = target, low, high
    CALIB_TIMESTAMP = c.get("timestamp")
    print("Calibration loaded from {} ({}): target={} low={} high={}".format(
        CALIB_JSON, CALIB_TIMESTAMP or "?",
        TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH))


_load_calibration()


def current_calibration():
    return {"TARGET_INTENSITY": TARGET_INTENSITY,
            "LOST_LINE_LOW": LOST_LINE_LOW,
            "LOST_LINE_HIGH": LOST_LINE_HIGH,
            "timestamp": CALIB_TIMESTAMP}


# =====================================================================
# 2. STATE SPACE (fix #3)
#    Bins are centred on the target so "on the edge", "a bit dark",
#    "very dark", "a bit bright", "very bright" are each one state:
#
#      bin 0 .. N_SIDE-1      dark side  (0 = darkest)
#      bin N_SIDE             centre: |intensity - target| <= CENTRE_HALF_WIDTH
#      bin N_SIDE+1 .. 2N     bright side (last = brightest)
#
#    With the current calibration (8 / 15 / 23) this gives 5 bins of
#    2-3 intensity units each instead of 8 bins of ~1.9.
# =====================================================================
N_SIDE = 2
CENTRE_HALF_WIDTH = 1
N_BINS = 2 * N_SIDE + 1
CENTRE_BIN = N_SIDE

MODE_FWD_LEFT  = 0   # forward, sensor tracking the LEFT edge of the line
MODE_FWD_RIGHT = 1   # forward, sensor tracking the RIGHT edge of the line
MODE_REVERSE   = 2   # reversing along the edge
MODES = (MODE_FWD_LEFT, MODE_FWD_RIGHT, MODE_REVERSE)
MODE_NAMES = {MODE_FWD_LEFT: "FWD-L", MODE_FWD_RIGHT: "FWD-R", MODE_REVERSE: "REV"}


def is_lost(intensity):
    return intensity <= LOST_LINE_LOW or intensity >= LOST_LINE_HIGH


def discretize(intensity, mode):
    """Map the valid window (LOW, HIGH) onto N_BINS target-centred bins."""
    i = min(max(intensity, LOST_LINE_LOW + 1), LOST_LINE_HIGH - 1)
    err = i - TARGET_INTENSITY
    if abs(err) <= CENTRE_HALF_WIDTH:
        return (CENTRE_BIN, mode)
    if err < 0:
        inner = TARGET_INTENSITY - CENTRE_HALF_WIDTH - 1   # dark value next to centre
        span = max(inner - (LOST_LINE_LOW + 1) + 1, 1)     # dark values available
        k = min((inner - i) * N_SIDE // span, N_SIDE - 1)  # 0 = next to centre
        return (CENTRE_BIN - 1 - k, mode)
    inner = TARGET_INTENSITY + CENTRE_HALF_WIDTH + 1       # bright value next to centre
    span = max((LOST_LINE_HIGH - 1) - inner + 1, 1)
    k = min((i - inner) * N_SIDE // span, N_SIDE - 1)
    return (CENTRE_BIN + 1 + k, mode)


def bin_ranges():
    """{bin: (min_intensity, max_intensity)} under the current calibration."""
    ranges = {}
    for i in range(LOST_LINE_LOW + 1, LOST_LINE_HIGH):
        b = discretize(i, MODE_FWD_LEFT)[0]
        lo, hi = ranges.get(b, (i, i))
        ranges[b] = (min(lo, i), max(hi, i))
    return ranges


def calibration_problems():
    """Reasons the calibration cannot support the target-centred bins."""
    problems = []
    if not LOST_LINE_LOW < TARGET_INTENSITY < LOST_LINE_HIGH:
        problems.append("target {} is not between low {} and high {}".format(
            TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH))
    missing = [b for b in range(N_BINS) if b not in bin_ranges()]
    if missing:
        problems.append("contrast window too narrow: bins {} can never occur".format(
            missing))
    return problems


# =====================================================================
# 3. ACTIONS (identical to rl_line_follower.py)
# =====================================================================
BASE_SPEED = 20

FWD_ACTIONS = {
    0: (BASE_SPEED + 8,  2),
    1: (BASE_SPEED + 4,  BASE_SPEED - 5),
    2: (BASE_SPEED,      BASE_SPEED),
    3: (BASE_SPEED - 5,  BASE_SPEED + 4),
    4: (2,               BASE_SPEED + 8),
}
REV_ACTIONS = {
    0: (-(BASE_SPEED + 8), -2),
    1: (-(BASE_SPEED + 4), -(BASE_SPEED - 5)),
    2: (-BASE_SPEED,       -BASE_SPEED),
    3: (-(BASE_SPEED - 5), -(BASE_SPEED + 4)),
    4: (-2,                -(BASE_SPEED + 8)),
}
N_ACTIONS = 5
STRAIGHT_ACTION = 2
# Swaps the two wheel speeds of a forward action (fix #2): the steering
# that holds a left edge, seen in a mirror, is what holds a right edge.
MIRROR_ACTION = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}


def action_speeds(mode, action):
    table = REV_ACTIONS if mode == MODE_REVERSE else FWD_ACTIONS
    return table[action]


def steering(mode, action):
    """Signed turn amount (left minus right wheel speed); 0 = straight."""
    ls, rs = action_speeds(mode, action)
    return ls - rs


def mirror_state_action(state, action):
    """FWD-L (state, action) -> equivalent FWD-R (state, action).

    Same brightness bin, mirrored steering. (rl-new.py also flipped the
    bin index, which undid the mirror.)
    """
    return (state[0], MODE_FWD_RIGHT), MIRROR_ACTION[action]


# =====================================================================
# 4. LEARNING PARAMETERS
# =====================================================================
LEARNING_RATE = 0.15
DISCOUNT      = 0.9
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.93        # 0.93 ** 45 ~= 0.038 -> reaches EPSILON_MIN
SEED_EPSILON  = 0.3         # FWD-R exploration after seed_fr (fine-tune only)

DEFAULT_EPISODES     = 45
EPISODE_MAX_STEPS    = 400
MIN_TRAINED_EPISODES = 10   # a mode's own Q-values are used in `run` from here

STEP_SLEEP           = 0.02 # real control period is ~50-80 ms incl. sysfs I/O

# Reward shaping weights (same as rl_line_follower.py).
W_EDGE      = 1.0
W_PROGRESS  = 0.30
LOST_REWARD = -100.0

# =====================================================================
# 5. RECOVERY / SEARCH SETTINGS (identical to rl_line_follower.py)
# =====================================================================
SEARCH_TURN_SPEED   = 14
SEARCH_CREEP_SPEED  = 12
SEARCH_SWEEPS       = 6
SEARCH_BASE_TIME    = 1.0
EDGE_DEADBAND       = 4

# =====================================================================
# 6. OBSTACLE AVOIDANCE (identical to rl_line_follower.py)
# =====================================================================
OBSTACLE_PROXIMITY  = 25
DETOUR_TURN_SPEED   = 20
DETOUR_TURN_TIME    = 0.9
DETOUR_ARC_TIME     = 1.9

# =====================================================================
# HARDWARE -- opened lazily, so status / audit / table work without
# motors (and even off the brick).
# =====================================================================
left_motor = right_motor = color_sensor = ir_sensor = sound = buttons = None
SpeedPercent = None


def init_hardware():
    global left_motor, right_motor, color_sensor, ir_sensor, sound, buttons
    global SpeedPercent
    if left_motor is not None:
        return
    from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C
    from ev3dev2.motor import SpeedPercent as _SpeedPercent
    from ev3dev2.sensor import INPUT_1, INPUT_4
    from ev3dev2.sensor.lego import ColorSensor, InfraredSensor
    from ev3dev2.sound import Sound
    from ev3dev2.button import Button

    SpeedPercent = _SpeedPercent
    left_motor = LargeMotor(OUTPUT_B)
    right_motor = LargeMotor(OUTPUT_C)
    color_sensor = ColorSensor(INPUT_1)
    color_sensor.mode = 'COL-REFLECT'
    sound = Sound()
    buttons = Button()
    try:
        ir_sensor = InfraredSensor(INPUT_4)
    except Exception:
        ir_sensor = None
        print("WARNING: no IR sensor on port 4 -> obstacle avoidance disabled.")


# =====================================================================
# Q-TABLE ("brain") -- per-mode epsilon and episode counts (fix #1)
# =====================================================================
BRAIN_VERSION = 2


def new_brain():
    return {
        "version": BRAIN_VERSION,
        "n_bins": N_BINS,
        "q": {},
        "epsilon": dict((m, EPSILON_START) for m in MODES),
        "episodes": dict((m, 0) for m in MODES),
        "seeded_fr": False,
        "calibration": current_calibration(),
    }


def load_brain(quiet=False):
    if not os.path.exists(Q_TABLE_PATH):
        return new_brain()
    with open(Q_TABLE_PATH, "rb") as f:
        brain = pickle.load(f)
    if not isinstance(brain, dict) or brain.get("version") != BRAIN_VERSION:
        raise SystemExit("{} is not a version-{} table. Move it away; it will "
                         "not be overwritten.".format(Q_TABLE_PATH, BRAIN_VERSION))
    if brain.get("n_bins") != N_BINS:
        raise SystemExit("{} was built with {} bins, this file uses {}. Move it "
                         "away and retrain.".format(Q_TABLE_PATH, brain.get("n_bins"),
                                                    N_BINS))
    if not quiet:
        warn_calibration_drift(brain)
    return brain


def save_brain(brain):
    """Write via a temp file so a flat battery can't leave a half-written table."""
    brain["calibration"] = current_calibration()
    tmp = Q_TABLE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(brain, f)
    os.replace(tmp, Q_TABLE_PATH)


def warn_calibration_drift(brain):
    old = brain.get("calibration") or {}
    keys = ("TARGET_INTENSITY", "LOST_LINE_LOW", "LOST_LINE_HIGH")
    if any(old.get(k) != current_calibration()[k] for k in keys) and \
            any(brain["episodes"][m] for m in MODES):
        print("WARNING: Q-table was trained with target/low/high = {}/{}/{}, "
              "calibration.json now says {}/{}/{}. The bins moved; retrain or "
              "re-audit before trusting the policy.".format(
                  old.get("TARGET_INTENSITY"), old.get("LOST_LINE_LOW"),
                  old.get("LOST_LINE_HIGH"), TARGET_INTENSITY, LOST_LINE_LOW,
                  LOST_LINE_HIGH))


def get_q(q_table, state, action):
    return q_table.get((state, action), 0.0)


def best_action(q_table, state, evaluating=False):
    q_values = [get_q(q_table, state, a) for a in range(N_ACTIONS)]
    if evaluating and all(v == 0.0 for v in q_values):
        return STRAIGHT_ACTION
    max_q = max(q_values)
    return random.choice([a for a, v in enumerate(q_values) if v == max_q])


def mode_trained(brain, mode):
    return brain["episodes"][mode] >= MIN_TRAINED_EPISODES


def policy_source(brain, mode):
    """Which Q-values drive `mode` in run, or None if it has no policy."""
    if mode_trained(brain, mode):
        return MODE_NAMES[mode]
    if mode == MODE_FWD_RIGHT and mode_trained(brain, MODE_FWD_LEFT):
        return "mirrored FWD-L"
    return None


def policy_action(brain, intensity, mode):
    q_table = brain["q"]
    state = discretize(intensity, mode)
    if mode_trained(brain, mode):
        return best_action(q_table, state, evaluating=True)
    if mode == MODE_FWD_RIGHT and mode_trained(brain, MODE_FWD_LEFT):
        # fix #2: same bin under FWD-L, then mirror the action only.
        action = best_action(q_table, (state[0], MODE_FWD_LEFT), evaluating=True)
        return MIRROR_ACTION[action]
    return STRAIGHT_ACTION


# =====================================================================
# AUDIT (fix #3 check) -- works on this file's table and the legacy one
# =====================================================================
def _sign(x):
    return (x > 0) - (x < 0)


def audit_mode(q_table, mode, n_bins):
    """Return (verdict, detail). verdict: PASS / FAIL / UNTRAINED.

    Checks steering DIRECTION only (left / straight / right). Going from
    dark to bright, the turning bins must switch direction exactly once:
    a sharp turn next to a gentle one the same way is fine, "straight"
    anywhere is fine (reported as a note at the extremes), but a bin
    turning against its neighbours on the same side -- the fault in the
    old q_table.pkl -- fails.
    """
    visited, steer, greedy = [], [], []
    for b in range(n_bins):
        s = (b, mode)
        if any(get_q(q_table, s, a) != 0.0 for a in range(N_ACTIONS)):
            a = best_action(q_table, s, evaluating=True)
            visited.append(b)
            greedy.append(a)
            steer.append(_sign(steering(mode, a)))
    if not visited:
        return "UNTRAINED", "no Q-values for this mode"

    detail = "greedy per bin (dark->bright): " + " ".join(
        "b{}:a{}".format(b, a) for b, a in zip(visited, greedy))
    problems, notes = [], []
    missing = [b for b in range(n_bins) if b not in visited]
    if missing:
        problems.append("bins never visited: {}".format(missing))
    turning = [s for s in steer if s != 0]
    switches = sum(1 for x, y in zip(turning, turning[1:]) if x != y)
    if switches > 1:
        problems.append("steering direction flips back and forth across the bins")
    elif switches == 0:
        problems.append("dark and bright bins do not steer in opposite directions")
    if steer[0] == 0 or steer[-1] == 0:
        notes.append("note: an extreme bin drives straight (rarely visited?); "
                     "more episodes usually fix it")
    if problems:
        return "FAIL", "\n      ".join([detail] + problems + notes)
    return "PASS", "\n      ".join([detail] + notes)


def audit_table(q_table, n_bins, title):
    print("=== Audit: {} ===".format(title))
    verdicts = {}
    for mode in MODES:
        verdict, detail = audit_mode(q_table, mode, n_bins)
        verdicts[mode] = verdict
        print("  {:5s} {:9s} {}".format(MODE_NAMES[mode], verdict, detail))
    return verdicts


def cmd_audit():
    brain = load_brain()
    verdicts = audit_table(brain["q"], N_BINS, Q_TABLE_PATH)
    for mode in MODES:
        src = policy_source(brain, mode)
        print("  {:5s} episodes={:3d} eps={:.3f} -> run uses: {}".format(
            MODE_NAMES[mode], brain["episodes"][mode], brain["epsilon"][mode],
            src or "nothing (mode disabled)"))
    return verdicts


def load_legacy_q():
    """The old q_table.pkl (either format) as a plain q dict; read-only."""
    if not os.path.exists(LEGACY_Q_TABLE_PATH):
        return None, None
    with open(LEGACY_Q_TABLE_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "q" in data:
        return data["q"], data.get("epsilon")
    return data, None


def cmd_audit_legacy():
    q_table, epsilon = load_legacy_q()
    if q_table is None:
        print("No {} to audit.".format(LEGACY_Q_TABLE_PATH))
        return
    print("(read-only; this table is not changed. Old layout: {} equal bins "
          "over the valid window, shared epsilon = {})".format(
              LEGACY_N_BINS, "?" if epsilon is None else "{:.4f}".format(epsilon)))
    audit_table(q_table, LEGACY_N_BINS, LEGACY_Q_TABLE_PATH)


def print_q(q_table, n_bins, ranges=None):
    for mode in MODES:
        print("\n--- mode {} ---".format(MODE_NAMES[mode]))
        print("bin  intens. | " + " | ".join("  a{}  ".format(a)
                                           for a in range(N_ACTIONS)) + " | greedy")
        for b in range(n_bins):
            s = (b, mode)
            qs = [get_q(q_table, s, a) for a in range(N_ACTIONS)]
            r = (ranges or {}).get(b)
            label = "{:2d}-{:<2d}".format(r[0], r[1]) if r else "  -  "
            print("{:3d}  {:7s} | ".format(b, label)
                  + " | ".join("{:6.1f}".format(v) for v in qs)
                  + " |  a{}".format(best_action(q_table, s, evaluating=True)))


def cmd_table():
    brain = load_brain()
    print("{} | per-mode epsilon: {}".format(Q_TABLE_PATH, ", ".join(
        "{}={:.3f}".format(MODE_NAMES[m], brain["epsilon"][m]) for m in MODES)))
    print_q(brain["q"], N_BINS, bin_ranges())


def cmd_table_legacy():
    q_table, epsilon = load_legacy_q()
    if q_table is None:
        print("No {}.".format(LEGACY_Q_TABLE_PATH))
        return
    print("{} (read-only) | epsilon = {}".format(LEGACY_Q_TABLE_PATH, epsilon))
    print_q(q_table, LEGACY_N_BINS)


# =====================================================================
# SENSING / REWARD
# =====================================================================
def read_intensity():
    return color_sensor.reflected_light_intensity


def get_reward(intensity, mode, action):
    if is_lost(intensity):
        return LOST_REWARD
    edge_penalty = -W_EDGE * abs(intensity - TARGET_INTENSITY)
    ls, rs = action_speeds(mode, action)
    avg = (ls + rs) / 2.0
    progress = -avg if mode == MODE_REVERSE else avg
    return edge_penalty + W_PROGRESS * progress


# =====================================================================
# MOTOR CONTROL
# =====================================================================
def drive(ls, rs):
    left_motor.on(SpeedPercent(ls))
    right_motor.on(SpeedPercent(rs))


def apply_action(mode, action):
    drive(*action_speeds(mode, action))


def stop():
    left_motor.off()
    right_motor.off()


# =====================================================================
# LINE SEARCH -- same phases as rl_line_follower.py, but EDGE-AWARE.
# The old search always turned the FWD-L way on "dark" and accepted the
# first edge it crossed, so a right-edge follower that lost the line was
# put back on the LEFT edge. Training FWD-R with it quietly learned the
# left edge (found in simulation while fixing #1/#2).
#
# edge_side(mode) = +1 for the FWD-L geometry (line on the side a
# drive(+s, -s) pivot turns the sensor toward), -1 for the mirror image.
# REV shares the FWD-L geometry, as it always did.
# =====================================================================
def edge_side(mode):
    return -1 if mode == MODE_FWD_RIGHT else 1


def near_edge():
    return abs(read_intensity() - TARGET_INTENSITY) <= EDGE_DEADBAND


def _sweep(direction, duration, side):
    """Pivot for `duration` s; stop only on the edge of the wanted side.

    Pivoting with `direction * side > 0` moves the sensor toward the
    line on the wanted edge, so that edge is entered from the dark side;
    otherwise it is entered from the bright side. An edge entered the
    other way is the opposite edge of the line and is swept past.
    """
    drive(direction * SEARCH_TURN_SPEED, -direction * SEARCH_TURN_SPEED)
    came_from_dark = None
    t0 = time.time()
    while time.time() - t0 < duration:
        i = read_intensity()
        if abs(i - TARGET_INTENSITY) > EDGE_DEADBAND:
            came_from_dark = i < TARGET_INTENSITY
        elif came_from_dark is not None and \
                came_from_dark == (direction * side > 0):
            stop()
            return True
        time.sleep(0.02)
    stop()
    return False


def search_for_line(mode):
    side = edge_side(mode)
    stop()
    time.sleep(0.1)

    # Phase 1: dark -> turn toward the line, bright -> turn away from it.
    for _ in range(12):
        if near_edge():
            stop()
            return True
        direction = side if (TARGET_INTENSITY - read_intensity()) > 0 else -side
        drive(direction * SEARCH_TURN_SPEED, -direction * SEARCH_TURN_SPEED)
        time.sleep(0.1)
    stop()

    # Phases 2+3: expanding sweeps (toward the line first), with a
    # forward creep between pairs.
    for k in range(1, SEARCH_SWEEPS + 1):
        t = SEARCH_BASE_TIME * k
        if _sweep(+side, t, side):
            return True
        if _sweep(-side, 2 * t, side):
            return True
        if _sweep(+side, t, side):
            return True
        drive(SEARCH_CREEP_SPEED, SEARCH_CREEP_SPEED)
        time.sleep(0.5)          # the next sweeps decide which edge is found
        stop()
    return False


# =====================================================================
# OBSTACLE AVOIDANCE (same as rl_line_follower.py)
# =====================================================================
def obstacle_ahead():
    return ir_sensor is not None and ir_sensor.proximity < OBSTACLE_PROXIMITY


def avoid_obstacle(mode):
    sound.beep()
    stop()
    time.sleep(0.2)
    drive(-SEARCH_CREEP_SPEED, -SEARCH_CREEP_SPEED)
    time.sleep(2.0)
    stop()
    drive(-DETOUR_TURN_SPEED, DETOUR_TURN_SPEED)
    time.sleep(DETOUR_TURN_TIME)
    drive(int(DETOUR_TURN_SPEED * 1.4), int(DETOUR_TURN_SPEED * 0.6))
    time.sleep(DETOUR_ARC_TIME)
    stop()
    if not search_for_line(mode):
        print("Could not re-find the line after the detour.")
        return False
    sound.beep()
    return True


# =====================================================================
# TRAINING
# =====================================================================
def train(mode, n_episodes):
    problems = calibration_problems()
    if problems:
        print("Calibration unusable: " + "; ".join(problems))
        print("Run calibrate.py again before training.")
        return
    brain = load_brain()
    q_table = brain["q"]
    epsilon = brain["epsilon"][mode]
    print("Training {} | episodes={} | resume epsilon={:.3f} | done so far={} "
          "| Q entries={}".format(MODE_NAMES[mode], n_episodes, epsilon,
                                  brain["episodes"][mode], len(q_table)))

    for episode in range(n_episodes):
        stop()
        sound.beep()
        time.sleep(1.0)                      # reposition the robot on the edge
        intensity = read_intensity()
        if is_lost(intensity) and not search_for_line(mode):
            print("  not on the edge and could not find it -> episode skipped")
            continue
        state = discretize(read_intensity(), mode)
        total_reward, step = 0.0, 0

        for step in range(EPISODE_MAX_STEPS):
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                action = best_action(q_table, state)
            apply_action(mode, action)
            time.sleep(STEP_SLEEP)

            intensity = read_intensity()
            reward = get_reward(intensity, mode, action)
            total_reward += reward
            old_q = get_q(q_table, state, action)

            if is_lost(intensity):
                # Terminal update: no bootstrap from a next state.
                q_table[(state, action)] = old_q + LEARNING_RATE * (reward - old_q)
                stop()
                if not search_for_line(mode):
                    print("  lost line, could not realign -> episode ends")
                    break
                state = discretize(read_intensity(), mode)
                continue

            next_state = discretize(intensity, mode)
            next_max = max(get_q(q_table, next_state, a) for a in range(N_ACTIONS))
            q_table[(state, action)] = old_q + LEARNING_RATE * (
                reward + DISCOUNT * next_max - old_q)
            state = next_state

        stop()
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        brain["epsilon"][mode] = epsilon
        brain["episodes"][mode] += 1
        save_brain(brain)
        print("Episode {}/{} | steps={} | reward={:.1f} | eps={:.3f}".format(
            episode + 1, n_episodes, step + 1, total_reward, epsilon))

    stop()
    verdict, detail = audit_mode(q_table, mode, N_BINS)
    print("Done. {} audit: {} ({})".format(MODE_NAMES[mode], verdict, detail))
    if verdict != "PASS":
        print("Train {} more before moving on (see `status`).".format(
            MODE_NAMES[mode]))


def seed_fr():
    """Initialise FWD-R from the mirrored FWD-L policy (fixes #1 and #2)."""
    brain = load_brain()
    q_table = brain["q"]
    if not mode_trained(brain, MODE_FWD_LEFT):
        print("FWD-L has {} episodes (< {}). Run train_fl first.".format(
            brain["episodes"][MODE_FWD_LEFT], MIN_TRAINED_EPISODES))
        return False
    verdict, detail = audit_mode(q_table, MODE_FWD_LEFT, N_BINS)
    if verdict != "PASS":
        print("FWD-L fails its audit, so mirroring it would copy the fault:\n  "
              + detail + "\nRun more train_fl episodes first.")
        return False
    if brain["episodes"][MODE_FWD_RIGHT] > 0:
        print("FWD-R already has {} trained episodes; not overwriting them.".format(
            brain["episodes"][MODE_FWD_RIGHT]))
        return False

    copied = 0
    for b in range(N_BINS):
        for a in range(N_ACTIONS):
            key = ((b, MODE_FWD_LEFT), a)
            if key in q_table:
                q_table[mirror_state_action(key[0], a)] = q_table[key]
                copied += 1
    brain["epsilon"][MODE_FWD_RIGHT] = SEED_EPSILON
    brain["seeded_fr"] = True
    save_brain(brain)
    print("Seeded FWD-R with {} mirrored FWD-L values; epsilon set to {}.".format(
        copied, SEED_EPSILON))
    print(audit_mode(q_table, MODE_FWD_RIGHT, N_BINS)[1])
    print("Next: train_fr to fine-tune on the real right edge.")
    return True


# =====================================================================
# DEMO RUN
# =====================================================================
def run_trained(duration_sec):
    brain = load_brain()
    if policy_source(brain, MODE_FWD_LEFT) is None:
        print("FWD-L is not trained yet. Run `status` for the steps.")
        return
    verdicts = audit_table(brain["q"], N_BINS, Q_TABLE_PATH)
    for mode in MODES:
        src = policy_source(brain, mode)
        if src is None:
            print("  {} unavailable in this run (not trained).".format(MODE_NAMES[mode]))
        elif src == MODE_NAMES[mode] and verdicts[mode] != "PASS":
            print("  WARNING: {} fails its audit; expect poor following.".format(
                MODE_NAMES[mode]))

    mode = MODE_FWD_LEFT
    last_fwd = mode
    pressed_before = set()
    lost_count = 0
    print("Running {} ({}). LEFT/RIGHT btn = edge side, DOWN = reverse, "
          "UP = forward.".format(MODE_NAMES[mode], policy_source(brain, mode)))
    start = time.time()

    while time.time() - start < duration_sec:
        # --- live mode switching, once per button press ---
        pressed = set(name for name in ("left", "right", "down", "up")
                      if getattr(buttons, name))
        new_presses = pressed - pressed_before
        pressed_before = pressed
        wanted = None
        if "left" in new_presses:
            wanted = MODE_FWD_LEFT
        elif "right" in new_presses:
            wanted = MODE_FWD_RIGHT
        elif "down" in new_presses:
            wanted = MODE_REVERSE
        elif "up" in new_presses:
            wanted = last_fwd
        if wanted is not None:
            if policy_source(brain, wanted) is None:
                stop()
                print("{} is not trained -> staying in {}.".format(
                    MODE_NAMES[wanted], MODE_NAMES[mode]))
                sound.beep()
                sound.beep()
            else:
                mode = wanted
                if mode != MODE_REVERSE:
                    last_fwd = mode
                print("Mode {} ({}).".format(MODE_NAMES[mode],
                                             policy_source(brain, mode)))
                sound.beep()

        if mode != MODE_REVERSE and obstacle_ahead():
            print("Obstacle! Detouring...")
            if not avoid_obstacle(mode):
                break
            continue

        intensity = read_intensity()
        if is_lost(intensity):
            lost_count += 1
            print("Lost the line (#{}) -> searching...".format(lost_count))
            if not search_for_line(mode):
                print("Could not realign, stopping.")
                break
            continue

        apply_action(mode, policy_action(brain, intensity, mode))
        time.sleep(STEP_SLEEP)

    stop()
    print("Run finished. Lost the line {} time(s).".format(lost_count))


def test_motors():
    print("Testing motors...")
    drive(BASE_SPEED, BASE_SPEED)
    time.sleep(10.0)
    stop()
    print("Motors test complete.")


# =====================================================================
# STATUS -- the execution-order checklist
# =====================================================================
def cmd_status():
    def box(done):
        return "[x]" if done else "[ ]"

    calib_ok = os.path.exists(CALIB_JSON) and not calibration_problems()
    brain = load_brain()
    q_table = brain["q"]
    v = dict((m, audit_mode(q_table, m, N_BINS)[0]) for m in MODES)
    fl_done = mode_trained(brain, MODE_FWD_LEFT) and v[MODE_FWD_LEFT] == "PASS"
    fr_done = mode_trained(brain, MODE_FWD_RIGHT) and v[MODE_FWD_RIGHT] == "PASS"
    rev_done = mode_trained(brain, MODE_REVERSE) and v[MODE_REVERSE] == "PASS"

    print("Bins under current calibration: " + ", ".join(
        "b{}={}-{}".format(b, r[0], r[1]) for b, r in sorted(bin_ranges().items())))
    for p in calibration_problems():
        print("  problem: " + p)
    steps = [
        (calib_ok, "1 calibrate.py -> calibration.json ({})".format(
            CALIB_TIMESTAMP or "missing"), "python3 calibrate.py"),
        (fl_done, "4 train_fl  ({} ep, audit {})".format(
            brain["episodes"][MODE_FWD_LEFT], v[MODE_FWD_LEFT]), "train_fl"),
        (brain["seeded_fr"] or brain["episodes"][MODE_FWD_RIGHT] > 0,
         "5 seed_fr", "seed_fr"),
        (fr_done, "5 train_fr  ({} ep, audit {})".format(
            brain["episodes"][MODE_FWD_RIGHT], v[MODE_FWD_RIGHT]), "train_fr"),
        (rev_done, "6 train_rev ({} ep, audit {})".format(
            brain["episodes"][MODE_REVERSE], v[MODE_REVERSE]), "train_rev"),
    ]
    print("    2 audit_legacy / 3 test_motors: optional checks, any time")
    next_cmd = None
    for done, label, cmd in steps:
        print("{} {}".format(box(done), label))
        if not done and next_cmd is None:
            next_cmd = cmd
    print("run policies: " + ", ".join("{}={}".format(
        MODE_NAMES[m], policy_source(brain, m) or "none") for m in MODES))
    if next_cmd is None:
        print("Next: run [seconds]  (all modes trained and audited)")
    elif next_cmd.startswith("python3"):
        print("Next: " + next_cmd)
    else:
        print("Next: python3 {} {}".format(os.path.basename(__file__), next_cmd))


# =====================================================================
# ENTRY POINT
# =====================================================================
USAGE = """usage: python3 {0} <command> [n]
  status              checklist: what is done, what to run next
  audit               check this file's Q-table (no motors)
  audit_legacy        check the old q_table.pkl, read-only (no motors)
  table / table_legacy  print a Q-table (no motors)
  test_motors         drive straight for 10 s
  train_fl  [episodes]
  seed_fr             FWD-R := mirrored FWD-L (needs FWD-L audit PASS)
  train_fr  [episodes]
  train_rev [episodes]
  run       [seconds]
""".format(os.path.basename(__file__))

OFFLINE_COMMANDS = {
    "status": cmd_status,
    "audit": cmd_audit,
    "audit_legacy": cmd_audit_legacy,
    "table": cmd_table,
    "table_legacy": cmd_table_legacy,
    "seed_fr": seed_fr,
}
TRAIN_COMMANDS = {
    "train_fl": MODE_FWD_LEFT,
    "train_fr": MODE_FWD_RIGHT,
    "train_rev": MODE_REVERSE,
}


def main(argv):
    if not argv:
        argv = DEFAULT_COMMAND
    cmd = argv[0]
    try:
        arg = int(argv[1]) if len(argv) > 1 else None
    except ValueError:
        print(USAGE)
        return 1
    if arg is not None and arg <= 0:
        print("The count must be a positive number.")
        return 1

    if cmd in OFFLINE_COMMANDS:
        result = OFFLINE_COMMANDS[cmd]()
        return 1 if result is False else 0
    if cmd not in TRAIN_COMMANDS and cmd not in ("run", "test_motors"):
        print(USAGE)
        return 1

    init_hardware()
    try:
        if cmd in TRAIN_COMMANDS:
            train(TRAIN_COMMANDS[cmd], arg or DEFAULT_EPISODES)
        elif cmd == "run":
            run_trained(arg or 150)
        else:
            test_motors()
    finally:
        stop()                               # never leave the motors running
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
