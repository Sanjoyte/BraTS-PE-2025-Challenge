# Load this project's configuration into the current shell.
#
#   source scripts/peds_env.sh
#
# Exports everything defined in the repo-root `.env`, including the three
# variables the nnU-Net CLI needs (nnUNet_raw / nnUNet_preprocessed /
# nnUNet_results), so `nnUNetv2_*` commands work directly in this shell.
#
# Must be sourced, not executed — running it in a subshell would export
# into a process that immediately exits.

_peds_env_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

if [ ! -f "${_peds_env_root}/.env" ]; then
    echo "error: ${_peds_env_root}/.env not found." >&2
    echo "       Create it first:  cp .env.example .env  && edit the paths." >&2
    unset _peds_env_root
    return 1 2>/dev/null || exit 1
fi

# `set -a` marks every subsequent assignment for export. Values in .env are
# quoted, so paths containing spaces survive.
set -a
# shellcheck disable=SC1091
. "${_peds_env_root}/.env"
set +a

# Make `python -c "import peds"` work from anywhere in the repo.
export PYTHONPATH="${_peds_env_root}${PYTHONPATH:+:${PYTHONPATH}}"

echo "BraTS-PED environment loaded from ${_peds_env_root}/.env"
echo "  nnUNet_raw          = ${nnUNet_raw}"
echo "  nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "  nnUNet_results      = ${nnUNet_results}"

unset _peds_env_root
