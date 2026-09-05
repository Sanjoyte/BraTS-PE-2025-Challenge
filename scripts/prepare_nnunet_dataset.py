#!/usr/bin/env python3
"""Convert raw_dataset/BraTS-PEDs-v1 into the nnU-Net v2 raw layout.

BraTS-PED ships one folder per case with hyphenated modality suffixes::

    BraTS-PED-00001-000/
        BraTS-PED-00001-000-t1n.nii.gz
        BraTS-PED-00001-000-t1c.nii.gz
        BraTS-PED-00001-000-t2w.nii.gz
        BraTS-PED-00001-000-t2f.nii.gz
        BraTS-PED-00001-000-seg.nii.gz     (Training only)

nnU-Net v2 wants a flat layout with numeric channel suffixes::

    $nnUNet_raw/Dataset501_BraTSPED/
        imagesTr/BraTS-PED-00001-000_0000.nii.gz   ... _0003.nii.gz
        labelsTr/BraTS-PED-00001-000.nii.gz
        imagesTs/...
        dataset.json

The channel order (0000=t1n, 0001=t1c, 0002=t2w, 0003=t2f) is the one the
original authors' runner used; keeping it means their trained checkpoints and
ours agree on what each input channel means.

By default images are **symlinked**, not copied — the raw dataset is ~33 GB and
duplicating it is pure waste. Pass --copy if your filesystem or downstream tool
cannot follow symlinks.

Labels are written as nnU-Net *regions* so the network is trained directly on
the six overlapping structures BraTS-PED is scored on.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peds.config import CFG  # noqa: E402

from peds.labels import (  # noqa: E402
    CHANNEL_NAMES,
    CHANNEL_ORDER,
    DATASET_JSON_LABELS,
    REGIONS_CLASS_ORDER,
)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        # Relative symlink keeps the working dir relocatable.
        os.symlink(os.path.relpath(src, dst.parent), dst)


def convert_split(
    src_dir: Path,
    images_dir: Path,
    labels_dir: Path | None,
    copy: bool,
    limit: int | None,
) -> list[str]:
    """Convert one split; returns the case identifiers written."""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"missing split directory: {src_dir}")

    cases = sorted(p for p in src_dir.iterdir() if p.is_dir())
    if limit is not None:
        cases = cases[:limit]

    images_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for case in cases:
        cid = case.name
        missing = [
            m for m in CHANNEL_ORDER if not (case / f"{cid}-{m}.nii.gz").is_file()
        ]
        if missing:
            print(f"  ! skipping {cid}: missing modality {', '.join(missing)}")
            continue

        seg = case / f"{cid}-seg.nii.gz"
        if labels_dir is not None and not seg.is_file():
            print(f"  ! skipping {cid}: no segmentation")
            continue

        for modality, idx in CHANNEL_ORDER.items():
            link_or_copy(
                case / f"{cid}-{modality}.nii.gz",
                images_dir / f"{cid}_{idx:04d}.nii.gz",
                copy,
            )
        if labels_dir is not None:
            link_or_copy(seg, labels_dir / f"{cid}.nii.gz", copy)

        written.append(cid)

    return written


def write_dataset_json(dataset_dir: Path, n_training: int) -> None:
    payload = {
        "channel_names": CHANNEL_NAMES,
        "labels": DATASET_JSON_LABELS,
        "regions_class_order": REGIONS_CLASS_ORDER,
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "name": "BraTS-PED 2025",
        "description": (
            "BraTS 2025 Pediatric brain tumour segmentation. Region-based "
            "labels: WT/TC/ET/NET/CC/ED."
        ),
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(payload, indent=4) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", action="store_true",
                    help="Copy image files instead of symlinking them (uses ~33 GB more disk).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Convert only the first N cases of each split (smoke tests).")
    ap.add_argument("--no-validation", action="store_true",
                    help="Skip the unlabelled Validation split (imagesTs).")
    args = ap.parse_args()

    dataset_dir = CFG.dataset_dir
    print(f"raw dataset : {CFG.raw_dataset}")
    print(f"destination : {dataset_dir}")
    print(f"mode        : {'copy' if args.copy else 'symlink'}")
    print()

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training -> imagesTr/labelsTr")
    train_ids = convert_split(
        CFG.training_dir, dataset_dir / "imagesTr", dataset_dir / "labelsTr",
        args.copy, args.limit,
    )
    print(f"  {len(train_ids)} cases")

    if not args.no_validation:
        print(f"Validation -> imagesTs (unlabelled)")
        test_ids = convert_split(
            CFG.validation_dir, dataset_dir / "imagesTs", None, args.copy, args.limit
        )
        print(f"  {len(test_ids)} cases")

    write_dataset_json(dataset_dir, len(train_ids))
    print(f"\nwrote {dataset_dir / 'dataset.json'}")

    if not train_ids:
        print("\nERROR: no training cases were converted.", file=sys.stderr)
        return 1

    print("\nDone. Next: nnUNetv2_plan_and_preprocess "
          f"-d {CFG.dataset_id} --verify_dataset_integrity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
