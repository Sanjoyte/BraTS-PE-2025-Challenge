"""Probability-space ensembling of the three segmentation members.

Each member writes, per case, an nnU-Net style ``<case>.npz`` holding a
``probabilities`` array of shape ``(n_channels, Z, Y, X)`` plus a ``<case>.nii.gz``
carrying the reference geometry. This module averages those probability maps
and converts the mean back into a label map.

The conversion is done through nnU-Net's own ``LabelManager`` rather than a bare
``argmax``. That matters here: this project trains on *overlapping regions*, so
the per-channel outputs are independent sigmoids that do not sum to one and an
argmax would silently produce nonsense. ``LabelManager`` applies
``regions_class_order`` correctly, and still does the right thing for a plain
softmax multi-class setup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peds.config import CFG  # noqa: E402


def _label_manager(model_dir: Path):
    """Build a LabelManager from a trained nnU-Net results folder."""
    plans = load_json(str(model_dir / "plans.json"))
    dataset_json = load_json(str(model_dir / "dataset.json"))
    return PlansManager(plans).get_label_manager(dataset_json)


def case_ids_from(npz_dir: Path) -> list[str]:
    return sorted(p.name[: -len(".npz")] for p in npz_dir.glob("*.npz"))


def average_probabilities(case: str, npz_dirs: list[Path], weights: list[float] | None = None):
    """Mean probability map for one case across members.

    Returns ``(probabilities, reference_nifti_path)``.
    """
    if weights is None:
        weights = [1.0] * len(npz_dirs)
    if len(weights) != len(npz_dirs):
        raise ValueError("weights and npz_dirs must have the same length")

    total = None
    total_w = 0.0
    reference = None

    for npz_dir, w in zip(npz_dirs, weights):
        npz_path = npz_dir / f"{case}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"{npz_path} is missing — did that member run with --save_probabilities?"
            )
        prob = np.load(npz_path, allow_pickle=True)["probabilities"].astype(np.float32)

        if total is None:
            total = prob * w
        else:
            if prob.shape != total.shape:
                raise ValueError(
                    f"shape mismatch for {case}: {npz_dir.name} gave {prob.shape}, "
                    f"expected {total.shape}. All members must share one label scheme."
                )
            total += prob * w
        total_w += w

        ref = npz_dir / f"{case}.nii.gz"
        if reference is None and ref.is_file():
            reference = ref

    if reference is None:
        raise FileNotFoundError(f"no reference .nii.gz alongside any .npz for {case}")

    return total / total_w, reference


def model_ensemble(
    npz_base_dirs,
    output_dir,
    model_dir,
    case_ids=None,
    weights=None,
) -> list[str]:
    """Ensemble every case and write label maps into output_dir."""
    npz_dirs = [Path(d) for d in npz_base_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for d in npz_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"member prediction dir not found: {d}")

    label_manager = _label_manager(Path(model_dir))

    if case_ids is None:
        # Only ensemble cases every member actually predicted.
        common = set(case_ids_from(npz_dirs[0]))
        for d in npz_dirs[1:]:
            common &= set(case_ids_from(d))
        case_ids = sorted(common)

    if not case_ids:
        raise RuntimeError("no cases common to all members — nothing to ensemble")

    written = []
    for case in case_ids:
        prob, reference = average_probabilities(case, npz_dirs, weights)

        # (C, Z, Y, X) -> (Z, Y, X) label map, region-aware.
        seg = label_manager.convert_probabilities_to_segmentation(prob)
        seg = np.asarray(seg)

        # nnU-Net works z-first; NIfTI on disk is x-first.
        seg = np.transpose(seg, (2, 1, 0)).astype(np.uint8)

        ref_img = nib.load(reference)
        out_path = output_dir / f"{case}.nii.gz"
        nib.save(nib.Nifti1Image(seg, ref_img.affine, ref_img.header), out_path)
        written.append(case)
        print(f"  ensembled {case} -> {out_path}")

    print(f"\n{len(written)} cases written to {output_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True,
                    help="Directories holding each member's .npz probability maps.")
    ap.add_argument("--output", "-o", default=None,
                    help=f"Output dir (default: {CFG.predictions / 'ensemble'}).")
    ap.add_argument("--model-dir", required=True,
                    help="A trained nnU-Net results folder (the one holding plans.json "
                         "and dataset.json) — used for the label scheme.")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Per-member weights; defaults to a uniform average.")
    args = ap.parse_args()

    output = Path(args.output) if args.output else CFG.predictions / "ensemble"
    model_ensemble(args.members, output, args.model_dir, weights=args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
