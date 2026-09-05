#!/usr/bin/env bash
# Run any command under a hard memory cap.
#
#   bash scripts/capped.sh nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_1epoch
#
# This machine has 16 GB of RAM and had been crashing when a training or
# preprocessing run grew past it. A systemd scope with MemoryMax turns that
# whole-machine freeze into an ordinary process kill: the run dies with an OOM
# message, the desktop stays alive.
#
# Caps come from .env:
#   PEDS_MEM_MAX   hard ceiling for the job      (default 11G)
#   PEDS_SWAP_MAX  how much swap it may also use (default 2G)
#
# Leave headroom: the cap must cover the Python process *and* its dataloader
# workers, while the OS, desktop and CUDA host allocations live outside it.
#
# If systemd user scopes are unavailable the command still runs, uncapped, with
# a warning — the worker-count limits in .env are the fallback protection.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/peds_env.sh" >/dev/null

MEM_MAX="${PEDS_MEM_MAX:-11G}"
SWAP_MAX="${PEDS_SWAP_MAX:-2G}"

if [ "$#" -eq 0 ]; then
    echo "usage: bash scripts/capped.sh <command> [args...]" >&2
    exit 2
fi

if ! command -v systemd-run >/dev/null 2>&1 || \
   ! systemd-run --user --scope -p MemoryMax=1G --quiet true >/dev/null 2>&1; then
    echo "warning: systemd user scopes unavailable — running WITHOUT a hard memory cap." >&2
    echo "         Worker limits in .env still apply." >&2
    exec "$@"
fi

echo "[capped] MemoryMax=${MEM_MAX} MemorySwapMax=${SWAP_MAX}"
echo "[capped] \$ $*"
echo

# --scope runs the command in the foreground under a transient cgroup.
# MemorySwapMax stops the job from thrashing swap instead of failing fast.
exec systemd-run --user --scope --quiet \
    -p MemoryMax="${MEM_MAX}" \
    -p MemorySwapMax="${SWAP_MAX}" \
    -- "$@"
