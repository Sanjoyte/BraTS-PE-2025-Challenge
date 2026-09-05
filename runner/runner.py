"""Command builders and drivers for the three ensemble members.

The published container shipped only ``nnunet_runner``; ``model_runner`` called
it three times into the same output folder and the Swin/HFF equivalents were
never released, so the ensemble could not actually be produced. The command
builders for all three members live here now, each writing to its own directory
so ``ensemble.model_ensemble`` has one probability map per member to average.

Every path is derived from the project config (see ``peds/config.py``); nothing
is pinned to the challenge container's ``/input`` and ``/tmp`` layout.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peds.config import CFG  # noqa: E402


def maybe_make_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _predict_cmd(
    input_path,
    output_path,
    trainer: str,
    dataset_id,
    configuration: str = "3d_fullres",
    plans: str = "nnUNetPlans",
    fold: str = "ensemble",
    checkpoint: str = "best",
    gamma: str = "",
    save_npz: bool = True,
) -> list[str]:
    """Build one ``nnUNetv2_predict`` invocation.

    ``fold="ensemble"`` omits ``-f`` so nnU-Net averages every trained fold,
    which is what the original runner did.
    """
    input_path = Path(input_path)
    output_path = maybe_make_dir(output_path)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_path),
        "-o", str(output_path),
        "-d", str(dataset_id),
        "-tr", trainer,
        "-p", plans,
        "-c", configuration,
        "-chk", f"checkpoint_{checkpoint}.pth",
    ]

    if fold and fold != "ensemble":
        cmd += ["-f", str(fold)]

    # The authors' initialization-scale flag must match the value used at
    # training time, otherwise the checkpoint is loaded into a differently
    # built network.
    if gamma != "":
        cmd += ["-gamma", str(gamma)]

    if save_npz:
        cmd.append("--save_probabilities")

    return cmd


def nnunet_runner(input_path, output_path, save_npz: bool = True, **kw) -> list[str]:
    """Plain nnU-Net member, trained with the adjustable initialization scale."""
    kw.setdefault("trainer", CFG.nnunet_trainer)
    kw.setdefault("dataset_id", CFG.dataset_id)
    kw.setdefault("gamma", CFG.gamma)
    return _predict_cmd(input_path, output_path, save_npz=save_npz, **kw)


def swin_runner(input_path, output_path, save_npz: bool = True, **kw) -> list[str]:
    """Swin UNETR member.

    Runs through the same nnU-Net CLI: the backbone swap lives in the trainer
    class. The 1005-epoch variant is built for 128^3 patches, so it needs the
    matching plans identifier rather than the default ``nnUNetPlans``.
    """
    trainer = kw.pop("trainer", CFG.swin_trainer)
    if "1005" in trainer:
        kw.setdefault("plans", "nnUNetPlansPatch")
    kw.setdefault("dataset_id", CFG.dataset_id)
    # The Swin trainer does its own initialization; gamma does not apply.
    kw.setdefault("gamma", "")
    return _predict_cmd(input_path, output_path, trainer=trainer, save_npz=save_npz, **kw)


def hff_runner(input_path, output_path, checkpoint=None, save_npz: bool = True, **kw) -> list[str]:
    """HFF-Net member.

    HFF-Net is not an nnU-Net trainer, so it goes through this project's own
    whole-volume inference script, which writes probability maps in the same
    ``<case>.npz`` + ``<case>.nii.gz`` layout the other two members produce.
    """
    input_path = Path(input_path)
    output_path = maybe_make_dir(output_path)
    checkpoint = checkpoint or (CFG.hff_results / "checkpoints" / "hff_best.pth")

    cmd = [
        sys.executable,
        str(CFG.repo_root / "scripts" / "hff_predict.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--checkpoint", str(checkpoint),
    ]
    if save_npz:
        cmd.append("--save-probabilities")
    for key, value in kw.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    return cmd


def run(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def model_runner(
    nnunet_input=None,
    noskull_input=None,
    hff_input=None,
    output_root=None,
    members=("nnunet", "swin", "hff"),
    dry_run: bool = False,
    **member_kwargs,
) -> dict[str, Path]:
    """Run each requested member into its own output directory.

    The plain nnU-Net member consumes skull-on images; Swin and HFF consume the
    skull-stripped ones, matching how the original authors trained them.

    Extra keyword arguments (e.g. ``trainer=...``, ``checkpoint=...``) are
    forwarded to each member's command builder, so a caller can override what
    the config file says without editing ``.env``.

    Returns a mapping of member name -> output directory.
    """
    output_root = Path(output_root) if output_root else CFG.predictions
    nnunet_input = Path(nnunet_input) if nnunet_input else CFG.dataset_dir / "imagesTs"
    noskull_input = Path(noskull_input) if noskull_input else nnunet_input

    # HFF does not read the flat nnU-Net layout: it needs the staged tree of
    # per-case folders holding the DTCWT/NSCT frequency volumes, which
    # scripts/prepare_hff_dataset.py builds under $PEDS_HFF_DATA.
    hff_input = hff_input if hff_input else CFG.hff_data / "Validation"

    builders = {
        "nnunet": (nnunet_runner, nnunet_input),
        "swin": (swin_runner, noskull_input),
        "hff": (hff_runner, hff_input),
    }

    produced: dict[str, Path] = {}
    for member in members:
        if member not in builders:
            raise ValueError(f"unknown member {member!r}; expected one of {list(builders)}")
        builder, member_input = builders[member]
        out_dir = output_root / member
        run(builder(member_input, out_dir, save_npz=True, **member_kwargs),
            dry_run=dry_run)
        produced[member] = out_dir

    return produced


def fast_model_runner(input_path=None, output_path=None, dry_run: bool = False) -> Path:
    """Single-model fast path: Swin UNETR only, labels straight out, no ensembling.

    Useful when a full three-member run is too slow and an approximate
    segmentation is enough.
    """
    input_path = Path(input_path) if input_path else CFG.dataset_dir / "imagesTs"
    output_path = maybe_make_dir(output_path or CFG.predictions / "swin_fast")

    run(swin_runner(input_path, output_path, save_npz=False), dry_run=dry_run)

    # nnU-Net drops bookkeeping JSON next to the predictions; drop it so the
    # folder contains only segmentations.
    for junk in output_path.glob("*.json"):
        junk.unlink()

    return output_path
