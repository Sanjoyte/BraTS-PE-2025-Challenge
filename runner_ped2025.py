#!/usr/bin/env python3
"""End-to-end inference: prepare inputs -> run each member -> ensemble.

This replaces the original challenge-container entry point, which was pinned to
``/input``, ``/tmp/inputs``, ``/tmp/outputs`` and ``/output``. Every location is
now taken from the project config or the command line, so the same script runs
on any machine.

Typical use, once the members are trained::

    python runner_ped2025.py --input raw_dataset/BraTS-PEDs-v1/Validation

Stages:

1. **prepare**  — copy/rename each case's four modalities into the flat
   ``<case>_0000..0003.nii.gz`` layout nnU-Net expects.
2. **strip**    — optional skull stripping (nnU-Net v1, Task070_autosegm).
   BraTS-PED is distributed already skull-stripped, so this is off by default;
   turn it on only for data that still has a skull.
3. **predict**  — each ensemble member writes probability maps to its own dir.
4. **ensemble** — average the probabilities and write final label maps.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from peds.config import CFG
from peds.labels import CHANNEL_ORDER
from runner.runner import model_runner
from ensemble.ensemble import model_ensemble
from skull_stripping.skull_stripping import skull_stripping

# BraTS modality suffix -> nnU-Net v2 channel filename. Derived from the shared
# channel order so training and inference can never disagree on what each input
# channel means.
NNUNET_V2_RENAME = {
    f"-{modality}.nii.gz": f"_{idx:04d}.nii.gz"
    for modality, idx in CHANNEL_ORDER.items()
}


def maybe_make_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def move_files(inputs_base_dir, outputs_base_dir) -> int:
    """Flatten per-case folders into one dir with numeric channel suffixes."""
    inputs_base_dir = Path(inputs_base_dir)
    outputs_base_dir = maybe_make_dir(outputs_base_dir)

    n = 0
    for case_dir in sorted(p for p in inputs_base_dir.iterdir() if p.is_dir()):
        case_name = case_dir.name
        missing = [s for s in NNUNET_V2_RENAME if not (case_dir / f"{case_name}{s}").is_file()]
        if missing:
            print(f"  ! skipping {case_name}: missing {', '.join(missing)}")
            continue
        for suffix, channel in NNUNET_V2_RENAME.items():
            shutil.copy2(case_dir / f"{case_name}{suffix}",
                         outputs_base_dir / f"{case_name}{channel}")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", default=None,
                    help="Dir of per-case folders (default: raw_dataset Validation split).")
    ap.add_argument("--output", "-o", default=None,
                    help="Where the final ensembled segmentations go "
                         "(default: $PEDS_PREDICTIONS/ensemble).")
    ap.add_argument("--work", default=None,
                    help="Scratch dir for prepared inputs (default: $PEDS_PREDICTIONS/_inputs).")
    ap.add_argument("--members", nargs="+", default=["nnunet", "swin", "hff"],
                    help="Which ensemble members to run.")
    ap.add_argument("--model-dir", default=None,
                    help="Trained nnU-Net results folder supplying the label scheme. "
                         "Defaults to the plain nnU-Net member's folder.")
    ap.add_argument("--skull-strip", action="store_true",
                    help="Run nnU-Net v1 skull stripping first. BraTS-PED already ships "
                         "skull-stripped, so leave this off for the challenge data.")
    ap.add_argument("--trainer", default=None,
                    help="Override the nnU-Net trainer class for this run "
                         "(otherwise PEDS_NNUNET_TRAINER from .env).")
    ap.add_argument("--checkpoint", default=None, choices=["best", "final"],
                    help="Which checkpoint to load (default: best).")
    ap.add_argument("--fold", default=None,
                    help="Fold to predict with. Default 'ensemble' averages every "
                         "trained fold; pass e.g. 0 when only one fold exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running them.")
    args = ap.parse_args()

    input_dir = Path(args.input) if args.input else CFG.validation_dir
    output_dir = Path(args.output) if args.output else CFG.predictions / "ensemble"
    work_dir = Path(args.work) if args.work else CFG.predictions / "_inputs"

    if not input_dir.is_dir():
        print(f"error: input dir not found: {input_dir}", file=sys.stderr)
        return 1

    print(f"input     : {input_dir}")
    print(f"work      : {work_dir}")
    print(f"output    : {output_dir}")
    print(f"members   : {', '.join(args.members)}")

    # 1. prepare -----------------------------------------------------------
    skull_dir = work_dir / "skull"
    print("\n[1/4] preparing inputs")
    n = move_files(input_dir, skull_dir)
    print(f"  {n} cases -> {skull_dir}")
    if n == 0:
        print("error: no usable cases found", file=sys.stderr)
        return 1

    # 2. skull stripping (optional) ---------------------------------------
    noskull_dir = skull_dir
    if args.skull_strip:
        noskull_dir = work_dir / "noskull"
        print("\n[2/4] skull stripping")
        skull_stripping(inputs_base_dir=input_dir, outputs_base_dir=noskull_dir)
    else:
        print("\n[2/4] skull stripping skipped (input already skull-stripped)")

    # 3. per-member prediction --------------------------------------------
    print("\n[3/4] running members")
    overrides = {}
    if args.trainer:
        overrides["trainer"] = args.trainer
    if args.checkpoint:
        overrides["checkpoint"] = args.checkpoint
    if args.fold:
        overrides["fold"] = args.fold

    produced = model_runner(
        nnunet_input=skull_dir,
        noskull_input=noskull_dir,
        output_root=CFG.predictions,
        members=tuple(args.members),
        dry_run=args.dry_run,
        **overrides,
    )

    if args.dry_run:
        print("\n[4/4] ensemble skipped (--dry-run)")
        return 0

    # 4. ensemble ----------------------------------------------------------
    print("\n[4/4] ensembling")
    model_dir = args.model_dir
    if model_dir is None:
        gamma_suffix = f"__std_gamma{CFG.gamma}" if CFG.gamma != "" else ""
        model_dir = (CFG.nnunet_results / CFG.dataset_name
                     / f"{CFG.nnunet_trainer}__nnUNetPlans__3d_fullres{gamma_suffix}")

    model_ensemble(
        npz_base_dirs=[produced[m] for m in args.members],
        output_dir=output_dir,
        model_dir=model_dir,
    )

    print(f"\nDone. Final segmentations: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
