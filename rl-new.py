#!/usr/bin/env python3
"""
Q-learning line follower for LEGO Mindstorms EV3 (ev3dev2)
=========================================================
Single merged script for Assignment 02. Replaces test.py and
train_autorealign.py.

What it covers (mapped to the marking criteria):
  * Learned forward / left-turn / right-turn edge following   (RL)
  * Learned REVERSE edge following                            (RL)
  * Clockwise AND anticlockwise following (edge-side flag
    inside the state, both sides trained)                     (RL)
  * Obstacle avoidance with the IR sensor                     (hardcoded, allowed)
  * Re-finding the path after an avoidance detour             (hardcoded, allowed)
  * Smoothness: forward-progress reward term + gentler
    pivot actions + terminal-update fix

Usage (run on the brick over SSH):
  python3 rl_line_follower.py train_fl   [episodes]   # train: forward, follow LEFT edge
  python3 rl_line_follower.py train_fr   [episodes]   # train: forward, follow RIGHT edge
  python3 rl_line_follower.py train_rev  [episodes]   # train: reverse along the edge
  python3 rl_line_follower.py run        [seconds]    # demo with learned policy
  python3 rl_line_follower.py table                   # print the Q-table summary

During `run`, the brick buttons switch behaviour live:
  LEFT button  -> forward mode, follow LEFT edge  (use for one lap direction)
  RIGHT button -> forward mode, follow RIGHT edge (use for the other direction)
  DOWN button  -> reverse mode
  UP button    -> back to the last forward mode

Fix list vs the old scripts:
  1. Calibration constants now match calibration_log.txt (14 / 8 / 20).
  2. State bins cover ONLY the valid intensity window (no dead bins).
  3. Terminal Q-update no longer bootstraps from a (bogus) next state.
  4. Reward includes a forward-progress term so "spin on the edge"
     is no longer an optimal policy; sharp pivots softened.
  5. Epsilon decay matched to episode count and persisted with the
     Q-table so resumed training continues where it left off.
  6. Unvisited states during evaluation default to "straight"
     instead of a random never-tried action.
  7. Line search: creep-forward + expanding pivot sweep (works after
     an obstacle detour, not just a small drift off the edge).
  8. Reverse action set + reverse training mode.
  9. Edge-side flag in the state -> one Q-table handles CW and ACW.
"""

import json
import os
import pickle
import random
import sys
import time

from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_4
from ev3dev2.sensor.lego import ColorSensor, InfraredSensor
from ev3dev2.sound import Sound
from ev3dev2.button import Button

# =====================================================================
# 1. CALIBRATION
#    Defaults below come from calibration_log.txt (floor=2, line=44).
#    If calibration.json exists next to this script (written by the
#    new calibrate.py), it OVERRIDES these values automatically, so
#    recalibrating on demo day needs no code edits.
# =====================================================================
TARGET_INTENSITY = 21      # edge between white line and dark floor
LOST_LINE_LOW    = 9       # <= this  -> fully on dark floor (lost)
LOST_LINE_HIGH   = 34      # >= this  -> fully on white line (lost, for edge-follow)

CALIB_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "calibration.json")


def _load_calibration():
    global TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH
    try:
        with open(CALIB_JSON) as f:
            c = json.load(f)
        TARGET_INTENSITY = int(c["TARGET_INTENSITY"])
        LOST_LINE_LOW = int(c["LOST_LINE_LOW"])
        LOST_LINE_HIGH = int(c["LOST_LINE_HIGH"])
        print("Calibration loaded from {} ({}): target={} low={} high={}".format(
            CALIB_JSON, c.get("timestamp", "?"),
            TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH))
    except IOError:
        print("No calibration.json found -> using built-in defaults "
              "(target={} low={} high={}). Run calibrate.py!".format(
                  TARGET_INTENSITY, LOST_LINE_LOW, LOST_LINE_HIGH))
    except (KeyError, ValueError) as e:
        print("calibration.json is malformed ({}) -> using defaults.".format(e))


_load_calibration()

# =====================================================================
# 2. STATE SPACE
#    Discretize ONLY the valid window [LOW, HIGH] so every bin is a
#    reachable, non-terminal state. 8 bins x 4 modes = 32 states.
# =====================================================================
N_BINS = 8
EPISODES = 100

# Modes (part of the state, so one Q-table learns all behaviours).
MODE_FWD_LEFT  = 0   # forward, sensor tracking the LEFT edge of the line
MODE_FWD_RIGHT = 1   # forward, sensor tracking the RIGHT edge of the line
MODE_REVERSE   = 2   # reversing along the edge
MODES = (MODE_FWD_LEFT, MODE_FWD_RIGHT, MODE_REVERSE)
MODE_NAMES = {MODE_FWD_LEFT: "FWD-L", MODE_FWD_RIGHT: "FWD-R", MODE_REVERSE: "REV"}

# =====================================================================
# 3. ACTIONS
#    Softer pivots than before (the old -10 counter-rotation at these
#    speeds was violent -> jerky line following). Reverse set mirrors
#    the forward set with negative speeds; the agent LEARNS reverse
#    steering itself (nothing is hardcoded about which action to pick).
# =====================================================================
BASE_SPEED = 20

FWD_ACTIONS = {
    0: (BASE_SPEED + 8,  2),                # sharp left
    1: (BASE_SPEED + 4,  BASE_SPEED - 5),   # left arc
    2: (BASE_SPEED,      BASE_SPEED),       # straight
    3: (BASE_SPEED - 5,  BASE_SPEED + 4),   # right arc
    4: (2,               BASE_SPEED + 8),   # sharp right
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
MIRROR_ACTION = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}


def action_speeds(mode, action):
    table = REV_ACTIONS if mode == MODE_REVERSE else FWD_ACTIONS
    return table[action]


# =====================================================================
# 4. LEARNING PARAMETERS
# =====================================================================
LEARNING_RATE = 0.15
DISCOUNT      = 0.9
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
# 0.93 ** 45 ~= 0.038 -> epsilon actually reaches EPSILON_MIN within
# the default 45 episodes (the old 0.98 left 40% randomness at the end).
EPSILON_DECAY = 0.975

DEFAULT_EPISODES  = 45
EPISODE_MAX_STEPS = 400
STEP_SLEEP        = 0.02      # real control period is ~50-80 ms incl. sysfs I/O

# Reward shaping weights.
W_EDGE     = 1.0    # penalty per unit of |intensity - target|
W_PROGRESS = 0.30   # bonus per unit of average wheel speed in the
                    # intended travel direction (kills the spin-in-place
                    # degenerate policy, rewards smooth fast following)
LOST_REWARD = -100.0

# =====================================================================
# 5. RECOVERY / SEARCH SETTINGS
# =====================================================================
SEARCH_TURN_SPEED   = 14
SEARCH_CREEP_SPEED  = 12
SEARCH_SWEEPS       = 6       # expanding sweep pairs before giving up
SEARCH_BASE_TIME    = 1.0     # seconds of first sweep; grows each pair
EDGE_DEADBAND       = 4

# =====================================================================
# 6. OBSTACLE AVOIDANCE (hardcoded, as the assignment allows)
# =====================================================================
OBSTACLE_PROXIMITY  = 25      # ir.proximity below this -> obstacle ahead
DETOUR_BACKUP_SPEED = 15      # reverse a little first to clear the obstacle
DETOUR_BACKUP_TIME  = 0.8     # seconds of straight reversing before turning
DETOUR_TURN_SPEED   = 20
DETOUR_TURN_TIME    = 1.5     # ~90 deg pivot; tune on your robot 0.5
DETOUR_ARC_TIME     = 10.0     # longer arc -> wider berth around the obstacle 2.6
DETOUR_ARC_INNER    = 0.5     # inner-wheel speed factor  (smaller -> wider arc)
DETOUR_ARC_OUTER    = 1.5     # outer-wheel speed factor  (bigger  -> wider arc)

# =====================================================================
# HARDWARE
# =====================================================================
left_motor   = LargeMotor(OUTPUT_B)
right_motor  = LargeMotor(OUTPUT_C)
color_sensor = ColorSensor(INPUT_1)
color_sensor.mode = 'COL-REFLECT'
sound   = Sound()
buttons = Button()

try:
    ir_sensor = InfraredSensor(INPUT_4)
except Exception:
    ir_sensor = None
    print("WARNING: no IR sensor on port 4 -> obstacle avoidance disabled.")

Q_TABLE_PATH = "q_table.pkl"


# =====================================================================
# Q-TABLE  (persisted together with epsilon so resumed training
# continues its exploration schedule instead of resetting to 1.0)
# =====================================================================
def load_brain():
    if os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "q" in data:
            return data["q"], data.get("epsilon", EPSILON_START)
        return data, EPSILON_START          # old-format file: raw q dict
    return {}, EPSILON_START


def save_brain(q_table, epsilon):
    with open(Q_TABLE_PATH, "wb") as f:
        pickle.dump({"q": q_table, "epsilon": epsilon}, f)


def get_q(q_table, state, action):
    return q_table.get((state, action), 0.0)


def best_action(q_table, state, evaluating=False):
    q_values = [get_q(q_table, state, a) for a in range(N_ACTIONS)]
    # Fix #6: during evaluation, if this state was never visited,
    # go straight instead of picking an arbitrary untried action.
    if evaluating and all(v == 0.0 for v in q_values):
        return STRAIGHT_ACTION
    max_q = max(q_values)
    best = [a for a, v in enumerate(q_values) if v == max_q]
    return random.choice(best)


# =====================================================================
# STATE / REWARD
# =====================================================================
def read_intensity():
    return color_sensor.reflected_light_intensity


def is_lost(intensity):
    return intensity <= LOST_LINE_LOW or intensity >= LOST_LINE_HIGH


def discretize(intensity, mode):
    """Map the VALID window [LOW, HIGH] onto N_BINS bins (fix #2)."""
    i = min(max(intensity, LOST_LINE_LOW + 1), LOST_LINE_HIGH - 1)
    span = float(LOST_LINE_HIGH - LOST_LINE_LOW)
    b = int((i - LOST_LINE_LOW) / span * N_BINS)
    b = min(b, N_BINS - 1)
    return (b, mode)


def policy_state(intensity, mode):
    """Use the left-edge policy as a geometric reference for right mode."""
    state = discretize(intensity, mode)
    if mode == MODE_FWD_RIGHT:
        return (N_BINS - 1 - state[0], MODE_FWD_LEFT)
    return state


def policy_action(q_table, intensity, mode):
    """Select an RL action, mirroring its steering for right-edge travel."""
    action = best_action(q_table, policy_state(intensity, mode), evaluating=True)
    if mode == MODE_FWD_RIGHT:
        return MIRROR_ACTION[action]
    return action


def get_reward(intensity, mode, action):
    if is_lost(intensity):
        return LOST_REWARD
    edge_penalty = -W_EDGE * abs(intensity - TARGET_INTENSITY)
    ls, rs = action_speeds(mode, action)
    avg = (ls + rs) / 2.0
    # Progress is speed in the INTENDED direction of travel.
    progress = -avg if mode == MODE_REVERSE else avg
    return edge_penalty + W_PROGRESS * progress


# =====================================================================
# MOTOR CONTROL
# =====================================================================
def apply_action(mode, action):
    ls, rs = action_speeds(mode, action)
    left_motor.on(SpeedPercent(ls))
    right_motor.on(SpeedPercent(rs))


def drive(ls, rs):
    left_motor.on(SpeedPercent(ls))
    right_motor.on(SpeedPercent(rs))


def stop():
    left_motor.off()
    right_motor.off()


# =====================================================================
# LINE SEARCH (fix #7)
# Works both for a small drift off the edge AND after an obstacle
# detour where the line may be metres of arc away at a strange angle:
#   phase 1: quick pivot toward the target intensity (cheap, common case)
#   phase 2: expanding left/right pivot sweeps
#   phase 3: creep forward a little and sweep again
# Returns True when the sensor is back near the edge.
# =====================================================================
def near_edge():
    return abs(read_intensity() - TARGET_INTENSITY) <= EDGE_DEADBAND


def _sweep(direction, duration):
    """Pivot in place for `duration` sec; stop early if edge found."""
    drive(direction * SEARCH_TURN_SPEED, -direction * SEARCH_TURN_SPEED)
    t0 = time.time()
    while time.time() - t0 < duration:
        if near_edge():
            stop()
            return True
        time.sleep(0.02)
    stop()
    return False


def search_for_line():
    stop()
    time.sleep(0.1)

    # Phase 1: steer toward target based on current brightness error.
    for _ in range(12):
        if near_edge():
            stop()
            return True
        direction = 1 if (TARGET_INTENSITY - read_intensity()) > 0 else -1
        drive(direction * SEARCH_TURN_SPEED, -direction * SEARCH_TURN_SPEED)
        time.sleep(0.1)
    stop()

    # Phases 2+3: expanding sweeps, with a backward creep between pairs.
    for k in range(1, SEARCH_SWEEPS + 1):
        t = SEARCH_BASE_TIME * k
        if _sweep(+1, t):            # sweep left
            return True
        if _sweep(-1, 2 * t):        # sweep right past the start point
            return True
        if _sweep(+1, t):            # re-centre
            return True
        drive(-SEARCH_CREEP_SPEED, -SEARCH_CREEP_SPEED)   # creep backward
        t0 = time.time()
        while time.time() - t0 < 0.5:
            if near_edge():
                stop()
                return True
            time.sleep(0.02)
        stop()
    
    # Phases 4+5: forward creep till line meets the sensor, then sweep again.
    for k in range(1, SEARCH_SWEEPS + 1):
        drive(SEARCH_CREEP_SPEED, SEARCH_CREEP_SPEED)   # creep forward
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if near_edge():
                stop()
                return True
            time.sleep(0.02)
        stop()
        t = SEARCH_BASE_TIME * k
        if _sweep(+1, t):            # sweep left
            return True
        if _sweep(-1, 2 * t):        # sweep right past the start point
            return True
        if _sweep(+1, t):            # re-centre
            return True

    return False


# =====================================================================
# OBSTACLE AVOIDANCE (hardcoded detour, then re-find the path)
# =====================================================================
def obstacle_ahead():
    return ir_sensor is not None and ir_sensor.proximity < OBSTACLE_PROXIMITY


def avoid_obstacle():
    sound.beep()
    stop()
    time.sleep(0.2)

    # 0) back up a little first: puts distance between the robot and
    #    the obstacle so the pivot + arc have room and the IR sensor
    #    isn't still staring at it point-blank after the turn.
    drive(-DETOUR_BACKUP_SPEED, -DETOUR_BACKUP_SPEED)
    time.sleep(DETOUR_BACKUP_TIME)
    stop()
    time.sleep(0.1)

    # 1) pivot right ~90 deg away from the obstacle
    drive(-DETOUR_TURN_SPEED, DETOUR_TURN_SPEED)
    time.sleep(DETOUR_TURN_TIME)

    # 2) wider, longer arc left around it. While arcing, watch the
    #    color sensor: if we happen to cross the line mid-arc, stop
    #    early instead of arcing past it.
    drive(int(DETOUR_TURN_SPEED * DETOUR_ARC_OUTER),
          int(DETOUR_TURN_SPEED * DETOUR_ARC_INNER))
    t0 = time.time()
    while time.time() - t0 < DETOUR_ARC_TIME:
        if near_edge():
            break
        time.sleep(0.02)
    stop()

    # 3) find the path again and resume RL policy control
    if not search_for_line():
        print("Could not re-find the line after the detour.")
        return False
    sound.beep()
    return True


# =====================================================================
# TRAINING (fixes #3, #4, #5)
# =====================================================================
def train(mode, n_episodes):
    q_table, epsilon = load_brain()
    print("Training mode {} | episodes={} | resume epsilon={:.3f} | Q entries={}".format(
        MODE_NAMES[mode], n_episodes, epsilon, len(q_table)))

    for episode in range(n_episodes):
        stop()
        sound.beep()
        time.sleep(1.0)                      # reposition the robot on the edge
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

            if is_lost(intensity):
                # Fix #3: TERMINAL update -> no bootstrap from any
                # "next state" (the old code leaked value back in).
                old_q = get_q(q_table, state, action)
                q_table[(state, action)] = old_q + LEARNING_RATE * (reward - old_q)
                stop()

                # Auto-realign and continue the same episode (keeps
                # hands-off training like train_autorealign.py did).
                if not search_for_line():
                    print("  lost line, could not realign -> episode ends")
                    break
                state = discretize(read_intensity(), mode)
                continue

            next_state = discretize(intensity, mode)
            old_q = get_q(q_table, state, action)
            next_max = max(get_q(q_table, next_state, a) for a in range(N_ACTIONS))
            q_table[(state, action)] = old_q + LEARNING_RATE * (
                reward + DISCOUNT * next_max - old_q)
            state = next_state

        stop()
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        save_brain(q_table, epsilon)         # fix #5: epsilon persisted
        print("Episode {}/{} | steps={} | reward={:.1f} | eps={:.3f}".format(
            episode + 1, n_episodes, step + 1, total_reward, epsilon))

    stop()
    print("Done. Q entries: {}".format(len(q_table)))


# =====================================================================
# DEMO RUN
# Greedy policy + obstacle avoidance + line re-finding.
# Brick buttons switch mode live (see header).
# =====================================================================

def run_trained(duration_sec):
    COUNT = 0
    q_table, _ = load_brain()
    if not q_table:
        print("No Q-table found. Train first.")
        return

    mode = MODE_FWD_RIGHT
    last_fwd = mode
    print("Running. LEFT/RIGHT btn = edge side, DOWN = reverse, UP = forward.")
    start = time.time()

    while time.time() - start < duration_sec:
        # --- live mode switching from the brick buttons ---
        if buttons.left:
            mode = last_fwd = MODE_FWD_LEFT
            sound.beep()
        elif buttons.right:
            mode = last_fwd = MODE_FWD_RIGHT
            sound.beep()
        elif buttons.down:
            mode = MODE_REVERSE
            sound.beep()
        elif buttons.up:
            print_table()
            sound.beep()
            sound.beep()

        # --- hardcoded obstacle avoidance (forward modes only) ---
        if mode != MODE_REVERSE and obstacle_ahead():
            print("Obstacle! Detouring...")
            if not avoid_obstacle():
                break
            continue

        intensity = read_intensity()
        if is_lost(intensity):
            COUNT += 1
            print("Lost the line (Count {COUNT}-> searching...")
            if not search_for_line():
                print("Could not realign, stopping.")
                break
            continue

        action = policy_action(q_table, intensity, mode)
        apply_action(mode, action)
        time.sleep(STEP_SLEEP)

    stop()


# =====================================================================
# INSPECTION
# =====================================================================
def print_table():
    q_table, epsilon = load_brain()
    print("epsilon = {:.3f}, entries = {}".format(epsilon, len(q_table)))
    for mode in MODES:
        print("\n--- mode {} ---".format(MODE_NAMES[mode]))
        print("bin | " + " | ".join("a{}".format(a) for a in range(N_ACTIONS)) + " | greedy")
        for b in range(N_BINS):
            s = (b, mode)
            qs = [get_q(q_table, s, a) for a in range(N_ACTIONS)]
            greedy = best_action(q_table, s, evaluating=True)
            print("{:3d} | ".format(b)
                  + " | ".join("{:6.1f}".format(v) for v in qs)
                  + " |  a{}".format(greedy))


# =====================================================================
# ENTRY POINT
# =====================================================================
USAGE = """usage:
  python3 rl_line_follower.py train_fl  [episodes]
  python3 rl_line_follower.py train_fr  [episodes]
  python3 rl_line_follower.py train_rev [episodes]
  python3 rl_line_follower.py run       [seconds]
  python3 rl_line_follower.py table
"""

if __name__ == "__main__":
    """if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    cmd = sys.argv[1]
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if cmd == "train_fl":
        train(MODE_FWD_LEFT, arg or DEFAULT_EPISODES)
    elif cmd == "train_fr":
        train(MODE_FWD_RIGHT, arg or DEFAULT_EPISODES)
    elif cmd == "train_rev":
        train(MODE_REVERSE, arg or DEFAULT_EPISODES)
    elif cmd == "run":
        run_trained(arg or 150)
    elif cmd == "table":
        print_table()
    else:
        print(USAGE)
        sys.exit(1)"""
    #train(MODE_FWD_LEFT, EPISODES)
    #train(MODE_FWD_RIGHT, EPISODES)
    #train(MODE_REVERSE, EPISODES)
    #print_table()
    run_trained(150)
