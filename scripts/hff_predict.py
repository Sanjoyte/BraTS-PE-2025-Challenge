#!/usr/bin/env python3
"""Whole-volume inference for HFF-Net, emitting ensemble-compatible outputs.

The upstream HFF repo ships `eval.py`, which only prints metrics: it needs
ground-truth masks, never writes predictions, and concatenates patch tensors
rather than reassembling volumes — so it cannot feed the ensemble. This script
is the missing inference half.

Two things it has to reconcile:

* **Geometry.** HFF works on a brain-centred 128^3 crop of a z-trimmed volume.
  Predictions are pasted back into the full original grid so every ensemble
  member shares one array shape.

* **Label space.** The nnU-Net members emit six *region* channels
  (WT/TC/ED/NET/CC/ET), while HFF emits per-class softmax over the five
  BraTS-PED labels. Class probabilities are summed into the region space, since
  a region's probability is the total mass of the labels composing it:

      P(WT) = p1+p2+p3+p4     P(NET) = p2
      P(TC) = p1+p2+p3        P(CC)  = p3
      P(ED) = p4              P(ET)  = p1

  That puts HFF on exactly the axes `ensemble.py` averages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peds.config import CFG  # noqa: E402

sys.path.insert(0, str(CFG.repo_root / "src" / "HFF"))

from peds.labels import (  # noqa: E402
    HIGH_FREQ_BANDS,
    MODALITIES,
    REGIONS,
    decode_regions,
)

# Region -> constituent class labels; ordering matches training exactly.
REGION_FROM_CLASSES = REGIONS

Z_TRIM = 3        # HFF's ThrowFirstZ drops the first three axial slices.
CROP = 128        # HFF's training/eval crop size.


def normalize(v: np.ndarray) -> np.ndarray:
    """HFF's Normalize: per-volume min-max rescale into [-1, 1]."""
    vmin, vmax = float(v.min()), float(v.max())
    rng = vmax - vmin
    if rng <= 0:
        return np.zeros_like(v, dtype=np.float32)
    return np.clip(2.0 * (v - vmin) / rng - 1.0, -1.0, 1.0).astype(np.float32)


def load_stack(case_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (low_freq (4,Z,Y,X), high_freq (16,Z,Y,X)) for one case."""
    import SimpleITK as sitk

    def read(name: str) -> np.ndarray:
        p = case_dir / f"{case_id}_{name}.nii.gz"
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} missing. Low-frequency volumes come from "
                f"scripts/prepare_hff_dataset.py; high-frequency ones from "
                f"scripts/run_nsct.sh (MATLAB)."
            )
        return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)

    low = np.stack([normalize(read(f"{m}_L")) for m in MODALITIES])
    high = np.stack([
        normalize(read(f"{m}_{b}")) for m in MODALITIES for b in HIGH_FREQ_BANDS
    ])
    return low, high


def brain_centre(volumes: np.ndarray) -> tuple[int, int, int]:
    """Centre of the non-background bounding box, as HFF's RandomCrop computes it."""
    mask = np.zeros(volumes.shape[1:], dtype=bool)
    for v in volumes:
        mask |= v != v[0, 0, 0]
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return tuple(s // 2 for s in volumes.shape[1:])  # type: ignore[return-value]
    return tuple(int((i.max() + i.min()) // 2) for i in idx)  # type: ignore[return-value]


def centred_slices(centre: tuple[int, int, int], shape: tuple[int, ...]):
    """Deterministic centred 128^3 window, clamped inside the volume."""
    out = []
    for c, extent in zip(centre, shape):
        lo = c - CROP // 2
        lo = max(0, min(lo, extent - CROP)) if extent >= CROP else 0
        out.append(slice(lo, lo + min(CROP, extent)))
    return tuple(out)


def classes_to_regions(class_prob: np.ndarray) -> np.ndarray:
    """(C,Z,Y,X) class softmax -> (6,Z,Y,X) region probabilities."""
    n_classes = class_prob.shape[0]
    regions = []
    for members in REGION_FROM_CLASSES.values():
        usable = [c for c in members if c < n_classes]
        if usable:
            regions.append(class_prob[usable].sum(axis=0))
        else:
            regions.append(np.zeros_like(class_prob[0]))
    return np.stack(regions).astype(np.float32)


@torch.no_grad()
def predict_case(model, case_dir: Path, case_id: str, device: torch.device) -> np.ndarray:
    """Region probabilities on the full original grid, shape (6, Z, Y, X)."""
    low, high = load_stack(case_dir, case_id)
    full_shape = low.shape[1:]

    # Match training preprocessing: drop the leading axial slices, then take the
    # brain-centred crop.
    low_t, high_t = low[:, Z_TRIM:], high[:, Z_TRIM:]
    sl = centred_slices(brain_centre(low_t), low_t.shape[1:])

    low_c = torch.from_numpy(low_t[:, sl[0], sl[1], sl[2]])[None].to(device)
    high_c = torch.from_numpy(high_t[:, sl[0], sl[1], sl[2]])[None].to(device)

    out_lf, out_hf, _, _ = model(low_c, high_c)
    # The dual-branch design trains both heads towards the same target, so the
    # ensemble prediction is their mean.
    prob = (torch.softmax(out_lf, dim=1) + torch.softmax(out_hf, dim=1)) / 2.0
    prob = prob[0].float().cpu().numpy()

    regions = classes_to_regions(prob)

    # Paste back onto the full grid; everything outside the crop is background.
    full = np.zeros((regions.shape[0],) + full_shape, dtype=np.float32)
    full[:, Z_TRIM:][:, sl[0], sl[1], sl[2]] = regions
    return full


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True,
                    help="Staged HFF split dir (one sub-folder per case).")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--checkpoint", required=True, help="Trained HFF .pth")
    ap.add_argument("--num-classes", type=int, default=5,
                    help="BraTS-PED has 5 label values (background + 4 structures).")
    ap.add_argument("--save-probabilities", action="store_true",
                    help="Write <case>.npz probability maps for ensembling.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    try:
        from model.HFF import HFFNet
    except ImportError as exc:
        raise SystemExit(f"cannot import HFF-Net from src/HFF ({exc})") from exc

    input_dir, output_dir = Path(args.input), Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HFFNet(len(MODALITIES),
                   len(MODALITIES) * len(HIGH_FREQ_BANDS),
                   args.num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    model.eval()

    cases = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not cases:
        print(f"error: no case folders under {input_dir}", file=sys.stderr)
        return 1

    for case_dir in cases:
        cid = case_dir.name
        prob = predict_case(model, case_dir, cid, device)

        seg = decode_regions(prob)

        # Reference geometry from one of this case's own modality volumes.
        ref = nib.load(str(case_dir / f"{cid}_{MODALITIES[0]}.nii.gz"))
        nib.save(nib.Nifti1Image(np.transpose(seg, (2, 1, 0)), ref.affine, ref.header),
                 output_dir / f"{cid}.nii.gz")
        if args.save_probabilities:
            np.savez_compressed(output_dir / f"{cid}.npz", probabilities=prob)
        print(f"  {cid} -> {output_dir / f'{cid}.nii.gz'}")

    print(f"\n{len(cases)} cases written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
