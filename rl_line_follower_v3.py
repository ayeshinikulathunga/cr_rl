#!/usr/bin/env python3
"""
Q-learning line follower v3 -- event-driven edition (ev3dev2)
=============================================================
Standalone file. It does not import or modify the other scripts and
never writes q_table.pkl or q_table_fixed.pkl. Its own table is
q_table_v3.pkl; it also READS q_table_working.pkl (the proven table,
see below) and calibration.json.

Why a v3: q_table_working.pkl (18 values) follows the line cleanly,
q_table_fixed.pkl (75 values) does not. The difference is the design,
not the amount of training:

  1. CLOSED-LOOP ACTIONS. A turn keeps turning until the light state
     CHANGES (with a timeout); forward drives until the state changes
     or FORWARD_TIME passes, speeding up while it stays on the edge.
     Every decision therefore has a clear outcome, and the agent only
     learns WHICH WAY to turn, not how long for. The fixed script's
     20 ms steps were mostly sensor noise, so its Q-values barely
     separated (margins 0.1 - 12) and the centre bin chose a sharp turn.
  2. THREE LIGHT STATES (BLACK / MIDDLE / WHITE) with a wide MIDDLE band
     around the calibrated target: 18 Q-values instead of 75.
  3. THE EDGE IS DETECTED, NOT CHOSEN. The transition caused by a turn
     tells which side the line is on (M_X / M_Y below), so there is no
     FWD-L / FWD-R button and no way to learn the wrong edge. After a
     detour or a lost line, a probe turn re-detects it.
  4. SIMPLE REWARD: +10 for landing in MIDDLE, -10 otherwise. No -100
     terminal and no episodes; exploration decays as exp(-steps/TEMP).
  5. Thresholds come from calibration.json (the working table's source
     used fixed 8 / 25, which on our mat would put "WHITE" at the edge).

Q-table file format: a Python literal (the format of micropython-lib's
pickle, which is what q_table_working.pkl is). Keys are
(mode, light_state, action_name) with mode True = line on the RIGHT of
the sensor (following the left edge, like FWD-L) and False = line on
the LEFT (right edge, like FWD-R). Both the plain dict and the wrapped
{"q": ..., "iterations": ...} form written here can be read.

Commands (python3 rl_line_follower_v3.py <command> [n]):
  status          thresholds, which table `run` uses, audit (no motors)
  table           print the table `run` would use (no motors)
  seed            q_table_v3.pkl := q_table_working.pkl, epsilon 0.2
  train [steps]   learn / fine-tune until epsilon < 0.01 (Ctrl-C saves)
  run [seconds]   greedy follower; uses q_table_v3.pkl, else the working one
During `run`: DOWN = drive in reverse, UP = forward again. The edge
side needs no button.

Written for the ev3dev Python 3.5 runtime (no f-strings).
"""

import ast
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON = os.path.join(HERE, "calibration.json")
Q_TABLE_PATH = os.path.join(HERE, "q_table_v3.pkl")
WORKING_Q_TABLE_PATH = os.path.join(HERE, "q_table_working.pkl")

DEFAULT_COMMAND = ["status"]

# =====================================================================
# 1. CALIBRATION (same file and defaults as rl_line_follower_fixed.py)
# =====================================================================
TARGET_INTENSITY = 21
LOST_LINE_LOW    = 9
LOST_LINE_HIGH   = 34
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


_load_calibration()

# MIDDLE = target +- this fraction of the calibrated window. Wider -> fewer
# turns but looser tracking; narrower -> tighter but more zig-zag. (tune)
EDGE_BAND_FRACTION = 0.25


def _thresholds():
    half = max(2, int(round((LOST_LINE_HIGH - LOST_LINE_LOW) * EDGE_BAND_FRACTION)))
    return TARGET_INTENSITY - half, TARGET_INTENSITY + half


BLACK_VALUE, WHITE_VALUE = _thresholds()   # <= BLACK_VALUE / >= WHITE_VALUE


def current_calibration():
    return {"TARGET_INTENSITY": TARGET_INTENSITY,
            "LOST_LINE_LOW": LOST_LINE_LOW,
            "LOST_LINE_HIGH": LOST_LINE_HIGH,
            "BLACK_VALUE": BLACK_VALUE,
            "WHITE_VALUE": WHITE_VALUE,
            "timestamp": CALIB_TIMESTAMP}


def calibration_problems():
    problems = []
    if not BLACK_VALUE < TARGET_INTENSITY < WHITE_VALUE:
        problems.append("target {} is not between BLACK {} and WHITE {}".format(
            TARGET_INTENSITY, BLACK_VALUE, WHITE_VALUE))
    if WHITE_VALUE - BLACK_VALUE < 4:
        problems.append("contrast window too narrow ({}..{})".format(
            BLACK_VALUE, WHITE_VALUE))
    return problems


# =====================================================================
# 2. STATES, ACTIONS, EDGE DETECTION
# =====================================================================
BLACK, MIDDLE, WHITE = "BLACK", "MIDDLE", "WHITE"
LIGHT_STATES = (BLACK, MIDDLE, WHITE)
FORWARD, TURN_LEFT, TURN_RIGHT = "forward", "turn_left", "turn_right"
ACTIONS = (FORWARD, TURN_LEFT, TURN_RIGHT)   # ties -> the earliest (forward)
MODES = (True, False)
MODE_NAMES = {True: "line on right (left edge)", False: "line on left (right edge)"}

# (before, action, after) transitions that reveal the edge side.
# Turning right into WHITE / out of BLACK means the line is on the right.
M_X = set([(MIDDLE, TURN_RIGHT, WHITE), (WHITE, TURN_LEFT, MIDDLE),
           (MIDDLE, TURN_LEFT, BLACK), (BLACK, TURN_RIGHT, MIDDLE)])   # -> True
M_Y = set([(MIDDLE, TURN_RIGHT, BLACK), (BLACK, TURN_LEFT, MIDDLE),
           (MIDDLE, TURN_LEFT, WHITE), (WHITE, TURN_RIGHT, MIDDLE)])   # -> False

# What a correct table must choose; follows from the M_X / M_Y geometry.
EXPECTED_POLICY = {
    True:  {BLACK: TURN_RIGHT, MIDDLE: FORWARD, WHITE: TURN_LEFT},
    False: {BLACK: TURN_LEFT,  MIDDLE: FORWARD, WHITE: TURN_RIGHT},
}


def update_mode(mode, before, action, after):
    t = (before, action, after)
    if t in M_X:
        return True
    if t in M_Y:
        return False
    return mode


# =====================================================================
# 3. LEARNING PARAMETERS (as in the working table's source)
# =====================================================================
ALPHA = 0.1
GAMMA = 0.9
TEMP = 1000.0          # epsilon = exp(-steps / TEMP)
EPSILON_STOP = 0.01    # training ends here (~4600 steps from scratch)
SEED_EPSILON = 0.2     # `seed` resumes at this epsilon (fine-tune only)
REWARD_EDGE = 10.0
REWARD_OFF = -10.0
SAVE_EVERY = 50        # steps between saves (not every step: flash is slow)


def epsilon(iterations):
    return math.exp(-iterations / TEMP)


# =====================================================================
# 4. MOTION SETTINGS (speed %, seconds; open-loop values -- tune)
# =====================================================================
FWD_MIN_SPEED = 18     # forward speed after a turn
FWD_MAX_SPEED = 30     # ramps up to this while staying on the edge
FWD_RAMP      = 2      # added per consecutive forward action
FORWARD_TIME  = 0.25
TURN_SPEED    = 11     # pivot component
TURN_BIAS     = 3      # small drive component, so a turn also creeps along
TURN_TIMEOUT  = 4.0    # a turn that never changes the state gives up here
POLL          = 0.01

CREEP_SPEED    = 12
CREEP_TIME     = 0.5
RECOVERY_TRIES = 4

OBSTACLE_PROXIMITY = 25
DETOUR_BACKUP_TIME = 2.0
DETOUR_TURN_SPEED  = 20
DETOUR_TURN_TIME   = 1.8   # ~180 deg pivot at DETOUR_TURN_SPEED (0.9 s was ~90) -- tune

# =====================================================================
# HARDWARE -- opened lazily so status / table work off the brick
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
# Q-TABLE I/O (Python-literal text, read with ast.literal_eval)
# =====================================================================
def new_q():
    return dict(((m, s, a), 0.0) for m in MODES for s in LIGHT_STATES for a in ACTIONS)


def read_table(path):
    """Return (q, iterations or None). Accepts the plain and wrapped formats."""
    with open(path) as f:
        data = ast.literal_eval(f.read().strip())
    if isinstance(data.get("q"), dict):
        raw, iterations = data["q"], data.get("iterations", 0)
    else:
        raw, iterations = data, None
    q = new_q()
    for key, value in raw.items():
        if key in q:
            q[key] = float(value)
    return q, iterations


def save_table(q, iterations):
    """Write via a temp file so a flat battery can't leave a half-written table."""
    data = {"version": 3, "q": q, "iterations": iterations,
            "calibration": current_calibration()}
    tmp = Q_TABLE_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(repr(data))
    os.replace(tmp, Q_TABLE_PATH)


def table_for_run():
    path = Q_TABLE_PATH if os.path.exists(Q_TABLE_PATH) else WORKING_Q_TABLE_PATH
    if not os.path.exists(path):
        return None, None
    return read_table(path)[0], path


def best_action(q, mode, light):
    best = ACTIONS[0]
    for a in ACTIONS[1:]:
        if q[(mode, light, a)] > q[(mode, light, best)]:
            best = a
    return best


def audit(q):
    """Print each greedy choice against EXPECTED_POLICY; True if all match."""
    ok = True
    for mode in MODES:
        parts = []
        for light in LIGHT_STATES:
            a = best_action(q, mode, light)
            vals = sorted((q[(mode, light, x)] for x in ACTIONS), reverse=True)
            good = a == EXPECTED_POLICY[mode][light]
            ok = ok and good
            parts.append("{}->{} (margin {:.1f}{})".format(
                light, a, vals[0] - vals[1], "" if good else ", WRONG"))
        print("  {:26s} {}".format(MODE_NAMES[mode], "; ".join(parts)))
    print("  audit: {}".format("PASS" if ok else "FAIL"))
    return ok


def print_table(q):
    print("{:26s} {:7s} | {:>9s} {:>9s} {:>10s} | greedy".format(
        "mode", "light", FORWARD, TURN_LEFT, TURN_RIGHT))
    for mode in MODES:
        for light in LIGHT_STATES:
            print("{:26s} {:7s} | {:9.2f} {:9.2f} {:10.2f} | {}".format(
                MODE_NAMES[mode], light, q[(mode, light, FORWARD)],
                q[(mode, light, TURN_LEFT)], q[(mode, light, TURN_RIGHT)],
                best_action(q, mode, light)))


# =====================================================================
# SENSING / MOTION
# =====================================================================
def light_state():
    i = color_sensor.reflected_light_intensity
    if i >= WHITE_VALUE:
        return WHITE
    if i <= BLACK_VALUE:
        return BLACK
    return MIDDLE


def drive(ls, rs):
    left_motor.on(SpeedPercent(ls))
    right_motor.on(SpeedPercent(rs))


def stop():
    left_motor.off()
    right_motor.off()


_forward_streak = 0


def do_action(action, before, direction=1, timeout=TURN_TIMEOUT):
    """Run one closed-loop action. Returns (light state after, timed_out).

    direction -1 drives in reverse; turns stay pivots, so they move the
    sensor the same way in both directions. Motors are left running
    between actions (the next action overrides them) for smooth motion.
    """
    global _forward_streak
    if action == FORWARD:
        speed = min(FWD_MIN_SPEED + FWD_RAMP * _forward_streak, FWD_MAX_SPEED)
        _forward_streak += 1
        drive(direction * speed, direction * speed)
        t0 = time.time()
        while time.time() - t0 < FORWARD_TIME:
            time.sleep(POLL)
            after = light_state()
            if after != before:
                return after, False
        return light_state(), False

    _forward_streak = 0
    turn = TURN_SPEED if action == TURN_RIGHT else -TURN_SPEED
    drive(direction * TURN_BIAS + turn, direction * TURN_BIAS - turn)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(POLL)
        after = light_state()
        if after != before:
            return after, False
    stop()
    return before, True


def find_edge(mode, direction=1):
    """Probe turns until the light state changes; the transition names
    the edge side. Returns (light, mode), or (None, mode) if not found."""
    for _ in range(RECOVERY_TRIES):
        for action, timeout in ((TURN_RIGHT, TURN_TIMEOUT), (TURN_LEFT, 2 * TURN_TIMEOUT)):
            before = light_state()
            after, timed_out = do_action(action, before, direction, timeout)
            if not timed_out:
                return after, update_mode(mode, before, action, after)
        drive(direction * CREEP_SPEED, direction * CREEP_SPEED)
        time.sleep(CREEP_TIME)
        stop()
    stop()
    return None, mode


# =====================================================================
# OBSTACLE AVOIDANCE (hardcoded, as in rl_line_follower_fixed.py)
# =====================================================================
def obstacle_ahead():
    return ir_sensor is not None and ir_sensor.proximity < OBSTACLE_PROXIMITY


def avoid_obstacle(mode):
    """Back up, then pivot ~180 deg toward the line's side so the sensor
    sweeps across the line and ends on the same edge, heading back.
    Returns the mode for the new heading: the line is now on the other
    side of the sensor. find_edge() afterwards confirms it."""
    sound.beep()
    stop()
    time.sleep(0.2)
    drive(-CREEP_SPEED, -CREEP_SPEED)
    time.sleep(DETOUR_BACKUP_TIME)
    stop()
    turn = DETOUR_TURN_SPEED if mode else -DETOUR_TURN_SPEED   # line on right -> turn right
    drive(turn, -turn)
    time.sleep(DETOUR_TURN_TIME)
    stop()
    return not mode


# =====================================================================
# TRAINING
# =====================================================================
def train(max_steps=None):
    problems = calibration_problems()
    if problems:
        print("Calibration unusable: " + "; ".join(problems))
        return
    if os.path.exists(Q_TABLE_PATH):
        q, iterations = read_table(Q_TABLE_PATH)
        iterations = iterations or 0
    else:
        q, iterations = new_q(), 0
    print("Training | steps so far={} | epsilon={:.3f} | BLACK<={} WHITE>={}".format(
        iterations, epsilon(iterations), BLACK_VALUE, WHITE_VALUE))
    if epsilon(iterations) < EPSILON_STOP and not max_steps:
        print("Already converged (epsilon < {}). Pass a step count to train more.".format(
            EPSILON_STOP))
        return

    sound.beep()
    light, mode = find_edge(True)
    if light is None:
        print("Could not find the edge. Place the robot on the line and retry.")
        return
    steps = 0
    try:
        while True:
            eps = epsilon(iterations)
            if max_steps is not None:
                if steps >= max_steps:
                    break
            elif eps < EPSILON_STOP:
                break
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = best_action(q, mode, light)

            after, timed_out = do_action(action, light)
            new_mode = update_mode(mode, light, action, after)
            reward = REWARD_EDGE if after == MIDDLE else REWARD_OFF
            next_max = max(q[(new_mode, after, a)] for a in ACTIONS)
            key = (mode, light, action)
            q[key] += ALPHA * (reward + GAMMA * next_max - q[key])
            iterations += 1
            steps += 1

            if iterations % SAVE_EVERY == 0:
                save_table(q, iterations)
                print("step {} | eps={:.3f} | {} | {} -{}-> {}".format(
                    iterations, eps, MODE_NAMES[new_mode], light, action, after))

            if timed_out:
                after, new_mode = find_edge(new_mode)
                if after is None:
                    print("Lost the line and could not recover -> stopping.")
                    break
            light, mode = after, new_mode
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        stop()
        save_table(q, iterations)
    print("Saved {} after {} total steps (epsilon {:.3f}).".format(
        Q_TABLE_PATH, iterations, epsilon(iterations)))
    audit(q)


def seed():
    """Start q_table_v3.pkl from the proven working table, for fine-tuning."""
    if os.path.exists(Q_TABLE_PATH):
        print("{} already exists; not overwriting it.".format(Q_TABLE_PATH))
        return False
    if not os.path.exists(WORKING_Q_TABLE_PATH):
        print("No {} to seed from.".format(WORKING_Q_TABLE_PATH))
        return False
    q = read_table(WORKING_Q_TABLE_PATH)[0]
    iterations = int(round(-TEMP * math.log(SEED_EPSILON)))
    save_table(q, iterations)
    print("Seeded {} from {}; training resumes at epsilon {}.".format(
        Q_TABLE_PATH, WORKING_Q_TABLE_PATH, SEED_EPSILON))
    return True


# =====================================================================
# RUN
# =====================================================================
def run(duration_sec):
    q, path = table_for_run()
    if q is None:
        print("No Q-table found ({} or {}).".format(Q_TABLE_PATH, WORKING_Q_TABLE_PATH))
        return
    print("Using {}".format(path))
    if not audit(q):
        print("WARNING: the table fails its audit; expect poor following.")

    direction = 1
    light, mode = find_edge(True, direction)
    if light is None:
        print("Could not find the edge. Place the robot on the line and retry.")
        return
    print("Following: {}. DOWN = reverse, UP = forward.".format(MODE_NAMES[mode]))
    pressed_before = set()
    lost_count = 0
    start = time.time()

    while time.time() - start < duration_sec:
        pressed = set(n for n in ("down", "up") if getattr(buttons, n))
        new_presses = pressed - pressed_before
        pressed_before = pressed
        if "down" in new_presses and direction == 1:
            direction = -1
            stop()
            sound.beep()
            print("Reverse.")
        elif "up" in new_presses and direction == -1:
            direction = 1
            stop()
            sound.beep()
            print("Forward.")

        if direction == 1 and obstacle_ahead():
            print("Obstacle! Turning around ({})...".format(
                "right" if mode else "left"))
            mode = avoid_obstacle(mode)
            light, mode = find_edge(mode, direction)
            if light is None:
                print("Could not re-find the line after the detour.")
                break
            sound.beep()
            print("Back on the line: {}.".format(MODE_NAMES[mode]))
            continue

        action = best_action(q, mode, light)
        after, timed_out = do_action(action, light, direction)
        mode = update_mode(mode, light, action, after)
        light = after
        if timed_out:
            lost_count += 1
            print("Lost the line (#{}) -> searching...".format(lost_count))
            light, mode = find_edge(mode, direction)
            if light is None:
                print("Could not realign, stopping.")
                break

    stop()
    print("Run finished. Lost the line {} time(s).".format(lost_count))


# =====================================================================
# STATUS / ENTRY POINT
# =====================================================================
def cmd_status():
    print("Calibration ({}): target={} low={} high={}".format(
        CALIB_TIMESTAMP or "defaults", TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH))
    print("Light states: BLACK <= {} < MIDDLE < {} <= WHITE".format(BLACK_VALUE, WHITE_VALUE))
    for p in calibration_problems():
        print("  problem: " + p)
    for path in (WORKING_Q_TABLE_PATH, Q_TABLE_PATH):
        if not os.path.exists(path):
            print("{}: missing".format(os.path.basename(path)))
            continue
        q, iterations = read_table(path)
        print("{}: {}".format(os.path.basename(path), "no step count" if iterations is None
                              else "{} steps, epsilon {:.3f}".format(
                                  iterations, epsilon(iterations))))
        audit(q)
    q, path = table_for_run()
    print("run uses: {}".format(os.path.basename(path) if path else "nothing"))


def cmd_table():
    q, path = table_for_run()
    if q is None:
        print("No Q-table found.")
        return False
    print(path)
    print_table(q)


USAGE = """usage: python3 {0} <command> [n]
  status          thresholds, tables and audits (no motors)
  table           print the table `run` would use (no motors)
  seed            q_table_v3.pkl := q_table_working.pkl (epsilon {1})
  train [steps]   learn until epsilon < {2}, or for n steps
  run [seconds]   follow the line (default 150 s)
""".format(os.path.basename(__file__), SEED_EPSILON, EPSILON_STOP)

OFFLINE_COMMANDS = {"status": cmd_status, "table": cmd_table, "seed": seed}


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
        return 1 if OFFLINE_COMMANDS[cmd]() is False else 0
    if cmd not in ("train", "run"):
        print(USAGE)
        return 1

    init_hardware()
    try:
        if cmd == "train":
            train(arg)
        else:
            run(arg or 150)
    finally:
        stop()                               # never leave the motors running
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
