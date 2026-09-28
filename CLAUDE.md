# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A tabular Q-learning edge-following line follower for a LEGO Mindstorms EV3 running **ev3dev** (`python-ev3dev2`). It is university coursework ("Assignment 02"). The RL policy handles forward, left/right-edge and reverse following. Obstacle avoidance and line re-finding are deliberately hardcoded, because the assignment allows that.

Hardware wiring is fixed in code: left motor `OUTPUT_B`, right motor `OUTPUT_C`, color sensor `INPUT_1` (reflected-light mode), IR sensor `INPUT_4` (optional; avoidance is disabled if it's missing).

## Running

Nothing here runs on the dev machine. Every script imports `ev3dev2` and touches sysfs devices at import time, so run them on the brick (usually over SSH) with `python3`. There is no build step, test suite or linter.

```
python3 calibrate.py            # sweep sensor over floor/line/edge, Ctrl-C to finish -> writes calibration.json
python3 rl_line_follower.py     # see note below about the entry point
```

**The CLI in `__main__` is disabled.** The argv dispatcher (`train_fl` / `train_fr` / `train_rev` / `run [seconds]` / `table`) is wrapped in a string literal. The block actually calls whichever of `train(...)`, `run_trained(...)`, `print_table()` or `test_motors()` is left uncommented at the bottom of the file. To change what runs, edit those lines, or restore the dispatcher if you're asked to. The module docstring still documents the CLI form.

Offline, you can inspect the Q-table without ev3dev:
```
python -c "import pickle; d=pickle.load(open('q_table.pkl','rb')); print(d['epsilon'], len(d['q']))"
```

## Script variants

There are three near-copies of the main script. Check which one the user means before editing:

- `rl_line_follower.py`: the committed "canonical" merged script, with `test_motors()`.
- `rl_line_follower-new.py`: an experiment. It retunes epsilon decay (0.975) and search speeds, creeps **backward** between search sweeps, and adds `DETOUR_BACKUP_*` / `DETOUR_ARC_*` constants. `run_trained(duration, mode)` takes the start mode. It keeps an alternative multi-leg `avoid_obstacle` inside a string literal.
- `rl-new.py`: the newest experiment. It adds `policy_state` / `policy_action` / `MIRROR_ACTION`, so **right-edge mode reuses the left-edge Q-values** (bin index flipped, action mirrored), and search gets extra forward-creep phases. The UP button prints the Q-table instead of returning to forward mode. It defaults to `MODE_FWD_RIGHT`.

The committed `q_table.pkl` only contains mode-0 (FWD-L) entries. That is why the mirroring approach in `rl-new.py` exists.

- `rl_line_follower_fixed.py`: a standalone corrected edition. It never writes `q_table.pkl`; its own table is `q_table_fixed.pkl` (v2 format: per-mode epsilon/episode counts, stored calibration). What it fixes:
  - The bins are centred on the target: 5 bins instead of 8.
  - Mirroring keeps the bin and mirrors only the action. rl-new.py's extra bin flip cancelled the mirror.
  - `search_for_line(mode)` is edge-aware. The old search always came back on the left edge, so FWD-R training learned the left edge.

  Its CLI works (`status` / `audit` / `audit_legacy` run off-brick, and hardware is opened lazily). `status` prints the ordered training checklist.

## Architecture (shared by all three variants)

- **Calibration handoff:** `calibrate.py` computes the floor and line levels from the 5th and 95th percentiles of the samples. It writes `TARGET_INTENSITY`, `LOST_LINE_LOW` and `LOST_LINE_HIGH` to `calibration.json` next to the script. The main script's `_load_calibration()` overrides its built-in defaults from that file at import time. Recalibrate by re-running `calibrate.py`, not by editing constants.
- **State** = `(bin, mode)`. `discretize()` maps only the valid window `(LOST_LINE_LOW, LOST_LINE_HIGH)` into `N_BINS` (8) bins. Mode is `MODE_FWD_LEFT` (0), `MODE_FWD_RIGHT` (1) or `MODE_REVERSE` (2). One Q-table covers all modes.
- **Actions:** 5 discrete wheel-speed pairs (sharp left, left arc, straight, right arc, sharp right). `REV_ACTIONS` mirrors `FWD_ACTIONS` with negative speeds, and `action_speeds(mode, a)` picks the right table.
- **Reward:** outside the valid window ("lost"), the reward is `LOST_REWARD` (-100). Otherwise it is `-W_EDGE*|intensity-target| + W_PROGRESS*speed_in_intended_direction`. The progress term exists to stop spin-in-place from becoming optimal.
- **Training loop (`train`):** epsilon-greedy. A lost state gets a *terminal* update with no bootstrap. After that, `search_for_line()` auto-realigns and the same episode continues. Epsilon decays once per episode and is saved **with** the Q-table.
- **Persistence format:** `q_table.pkl` = `{"q": {((bin, mode), action): value}, "epsilon": float}`. `load_brain()` also accepts the old format (a bare q dict). Training resumes from the saved epsilon.
- **Demo (`run_trained`):** greedy policy. States never visited fall back to `STRAIGHT_ACTION`. The brick buttons switch mode live. In forward modes, IR proximity below `OBSTACLE_PROXIMITY` triggers `avoid_obstacle()` (back up, pivot, arc, then `search_for_line()`). Losing the line triggers `search_for_line()`, which first steers toward the target, then does expanding pivot sweeps with creeps between them.

## Gotchas

- Target **Python 3.5** (the ev3dev stretch runtime). Use `.format()`, not f-strings, and avoid newer stdlib APIs. For example, `rl-new.py`'s `"Lost the line (Count {COUNT}..."` is missing `.format` and prints the braces literally.
- `calibration.json` is resolved relative to the script's directory, but `Q_TABLE_PATH = "q_table.pkl"` is relative to the **current working directory**. Run from the repo directory, or training will write a fresh table somewhere else.
- The "fix list" and calibration comments in the module docstring (e.g. "14 / 8 / 20") are stale. The in-code defaults are 21/9/34, and `calibration.json` overrides both.
- Detour and search timings (`DETOUR_*_TIME`, `SEARCH_*`) are open-loop, tuned on the physical robot and marked "tune". Changing them changes real-world behavior, and nothing here can verify that off-robot.
- Changing `N_BINS`, the action set or the state tuple invalidates the existing `q_table.pkl`.
