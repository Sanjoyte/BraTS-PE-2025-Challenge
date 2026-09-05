#!/usr/bin/env bash
# Run the HFF high-frequency (NSCT) decomposition over the staged HFF tree.
#
# Requires MATLAB with the Image Processing Toolbox (for imadjust/stretchlim)
# and, ideally, the Parallel Computing Toolbox — nsct_hf uses `parfor`, which
# degrades to a serial loop without it, just slower.
#
# Prerequisite: scripts/prepare_hff_dataset.py has staged the cases.
# Produces:     <case>_<modality>_H1..H4.nii.gz next to each staged modality.
#
#   bash scripts/run_nsct.sh [Training|Validation]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/peds_env.sh" >/dev/null

SPLIT="${1:-Training}"
BASE_DIR="${PEDS_HFF_DATA}/${SPLIT}"
TBX_DIR="${REPO_ROOT}/src/HFF/NSCT_BTS/nsct_toolbox"
NSCT_DIR="${REPO_ROOT}/src/HFF/NSCT_BTS"
MATLAB_BIN="${PEDS_MATLAB:-matlab}"

if ! command -v "${MATLAB_BIN}" >/dev/null 2>&1; then
    echo "error: MATLAB not found (tried '${MATLAB_BIN}')." >&2
    echo "       Install MATLAB + Image Processing Toolbox, or set PEDS_MATLAB in .env" >&2
    echo "       to the full path of the matlab executable." >&2
    exit 1
fi

if [ ! -d "${BASE_DIR}" ]; then
    echo "error: ${BASE_DIR} does not exist." >&2
    echo "       Run: python scripts/prepare_hff_dataset.py --split ${SPLIT}" >&2
    exit 1
fi

echo "NSCT high-frequency decomposition"
echo "  cases   : ${BASE_DIR}"
echo "  toolbox : ${TBX_DIR}"
echo

# The bundled toolbox ships prebuilt .mexa64 binaries for atrousc/zconv2, so no
# compilation step is needed on x86-64 Linux.
"${MATLAB_BIN}" -batch \
    "addpath('${NSCT_DIR}'); nsct_hf('${BASE_DIR}', '${TBX_DIR}', 'BraTS-PED-*'); exit"

echo
echo "Done. Verify with:"
echo "  python scripts/prepare_hff_dataset.py --split ${SPLIT} --check"
