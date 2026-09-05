#!/usr/bin/env python3
"""Build the HFF-Net input tree from raw_dataset/BraTS-PEDs-v1.

HFF-Net consumes each modality twice: a low-frequency volume (DTCWT) and four
directional high-frequency volumes (NSCT). Its dataloader addresses files with
**underscores** (`<case>_<modality>.nii.gz`), while BraTS-PED ships **hyphens**
(`<case>-t1c.nii.gz`) — mixing the two is the single most common way to get an
empty dataset, and the upstream HFF README calls it out explicitly.

This script does three things:

  1. Re-links each case into `$PEDS_HFF_DATA` under underscore naming.
  2. Runs the DTCWT low-frequency decomposition (pure Python).
  3. Emits the train/val split `.txt` lists `train.py` / `eval.py` expect.

The high-frequency (NSCT) half is MATLAB, so it is a separate step — run
`scripts/run_nsct.sh` after this. Use --check to report which cases are still
missing their `_H1.._H4` volumes.

Modality naming for BraTS-PED is `t1n / t1c / t2w / t2f`, i.e. the BraTS-23
convention that sits commented out in the upstream `train.py`, not the
BraTS-19/20 `flair / t1 / t1ce / t2` default.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peds.config import CFG  # noqa: E402
from peds.labels import HIGH_FREQ_BANDS, MODALITIES  # noqa: E402

# The --selected_modal list to pass to HFF train.py / eval.py for BraTS-PED:
# four low-frequency channels first, then sixteen high-frequency ones.
SELECTED_MODAL = (
    [f"{m}_L" for m in MODALITIES]
    + [f"{m}_{b}" for m in MODALITIES for b in HIGH_FREQ_BANDS]
)


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(os.path.relpath(src, dst.parent), dst)


def stage_cases(src_dir: Path, dest_root: Path, with_seg: bool) -> list[str]:
    """Mirror hyphen-named cases into underscore-named folders."""
    staged: list[str] = []
    for case in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        cid = case.name
        wanted = {m: case / f"{cid}-{m}.nii.gz" for m in MODALITIES}
        if any(not p.is_file() for p in wanted.values()):
            print(f"  ! skipping {cid}: missing a modality")
            continue
        seg = case / f"{cid}-seg.nii.gz"
        if with_seg and not seg.is_file():
            print(f"  ! skipping {cid}: no segmentation")
            continue

        out = dest_root / cid
        for m, p in wanted.items():
            link(p, out / f"{cid}_{m}.nii.gz")
        if seg.is_file():
            link(seg, out / f"{cid}_seg.nii.gz")
        staged.append(cid)
    return staged


def run_dtcwt(dest_root: Path, cases: list[str], nlevels: int = 1) -> None:
    """Low-frequency decomposition for every staged modality."""
    sys.path.insert(0, str(CFG.repo_root / "src" / "HFF"))
    try:
        from DTCWT_LF import process_nii_lowpass
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise SystemExit(
            f"cannot import HFF's DTCWT_LF ({exc}).\n"
            "Install its dependency first:  pip install dtcwt"
        ) from exc

    for i, cid in enumerate(cases, 1):
        case_dir = dest_root / cid
        for m in MODALITIES:
            src = case_dir / f"{cid}_{m}.nii.gz"
            out = case_dir / f"{cid}_{m}_L.nii.gz"
            if out.is_file():
                continue
            process_nii_lowpass(str(src), str(out), nlevels=nlevels)
        if i % 10 == 0 or i == len(cases):
            print(f"  DTCWT {i}/{len(cases)}")


def missing_high_freq(dest_root: Path, cases: list[str]) -> list[str]:
    out = []
    for cid in cases:
        for m in MODALITIES:
            for band in HIGH_FREQ_BANDS:
                if not (dest_root / cid / f"{cid}_{m}_{band}.nii.gz").is_file():
                    out.append(cid)
                    break
            else:
                continue
            break
    return out


def write_splits(dest_root: Path, cases: list[str], val_fraction: float, seed: int) -> None:
    """Write the newline-delimited case-folder lists HFF's loader reads."""
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val, train = shuffled[:n_val], shuffled[n_val:]

    for name, subset in (("train", train), ("val", val)):
        path = dest_root / f"{name}.txt"
        path.write_text(
            "\n".join(str(dest_root / cid) for cid in sorted(subset)) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {path} ({len(subset)} cases)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["Training", "Validation"], default="Training")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stage only the first N cases (smoke tests).")
    ap.add_argument("--skip-dtcwt", action="store_true",
                    help="Stage and split only; do not run the low-frequency transform.")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--check", action="store_true",
                    help="Only report which cases still lack NSCT high-frequency volumes.")
    args = ap.parse_args()

    src_dir = CFG.raw_dataset / args.split
    dest_root = CFG.hff_data / args.split
    dest_root.mkdir(parents=True, exist_ok=True)

    if args.check:
        cases = sorted(p.name for p in dest_root.iterdir() if p.is_dir())
        miss = missing_high_freq(dest_root, cases)
        print(f"{len(cases) - len(miss)}/{len(cases)} cases have all "
              f"{len(MODALITIES) * len(HIGH_FREQ_BANDS)} high-frequency volumes")
        if miss:
            print(f"missing NSCT output for {len(miss)} cases, e.g. {miss[:5]}")
            print("run: bash scripts/run_nsct.sh")
        return 0

    print(f"source      : {src_dir}")
    print(f"destination : {dest_root}\n")

    print("staging cases (hyphen -> underscore naming)")
    cases = stage_cases(src_dir, dest_root, with_seg=(args.split == "Training"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"  {len(cases)} cases staged")

    if not cases:
        print("error: nothing staged", file=sys.stderr)
        return 1

    if not args.skip_dtcwt:
        print("\nlow-frequency decomposition (DTCWT)")
        run_dtcwt(dest_root, cases)

    if args.split == "Training":
        print("\nsplits")
        write_splits(dest_root, cases, args.val_fraction, args.seed)

    miss = missing_high_freq(dest_root, cases)
    print(f"\nhigh-frequency (NSCT) volumes still missing for {len(miss)}/{len(cases)} cases")
    if miss:
        print("next: bash scripts/run_nsct.sh   (requires MATLAB + Image Processing Toolbox)")

    print("\n--selected_modal for HFF train.py / eval.py:")
    print("  " + " ".join(SELECTED_MODAL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
