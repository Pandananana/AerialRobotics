#!/usr/bin/env bash
# Run the racing assignment world headless: no rendering, fast mode, controller
# stdout/stderr piped to this terminal. Webots exits on its own once the
# controller calls simulationQuit(0) after lap 3.
set -euo pipefail

WEBOTS="${WEBOTS:-/Applications/Webots.app/Contents/MacOS/webots}"
WORLD="$(dirname "$0")/worlds/crazyflie_world_assignment.wbt"

exec "$WEBOTS" \
    --mode=fast \
    --no-rendering \
    --minimize \
    --batch \
    --stdout \
    --stderr \
    "$WORLD"
