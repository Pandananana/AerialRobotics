#!/usr/bin/env bash
# Run the PID-tuning exercise world headless. Streams the controller's
# step-response metrics (steady-state error, overshoot, rise time) to stdout
# instead of opening a matplotlib window, so the output can be parsed by an
# automated tuner. Webots exits on its own once the tuning sweep is done.
#
# Tune which loop is exercised by editing `self.tuning_level` in
# controllers/main/exercises/ex1_pid_control.py
# (one of: "vel_z", "pos_z", "vel_xy", "pos_xy", "att_rp", "att_y",
#  "rate_rp", "rate_y", or "off" to disable).
set -euo pipefail

export PID_TUNE_HEADLESS=1
export MPLBACKEND=Agg

WEBOTS="${WEBOTS:-/Applications/Webots.app/Contents/MacOS/webots}"
WORLD="$(dirname "$0")/worlds/crazyflie_world_excercise.wbt"

exec "$WEBOTS" \
    --mode=fast \
    --no-rendering \
    --minimize \
    --batch \
    --stdout \
    --stderr \
    "$WORLD"
