"""Single source of truth for every machine-specific path and setting.

Values are read from the environment, which is populated from the repo-root
`.env` file (see `.env.example`). Nothing here is hardcoded to one machine, so
the same checkout works on native Linux and under WSL2.

Usage::

    from peds.config import CFG
    print(CFG.nnunet_raw)

Importing this module also pushes the nnU-Net v2 variables (``nnUNet_raw``,
``nnUNet_preprocessed``, ``nnUNet_results``) into ``os.environ``, so the
nnU-Net CLI works in any subprocess spawned from Python without the caller
having to export them first.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# `.env` is authoritative: a stale `export nnUNet_raw=...` left over in a
# teammate's shell must not silently redirect the pipeline somewhere else.
# The exception is genuinely per-invocation settings, where the usual
# `CUDA_VISIBLE_DEVICES=1 python ...` idiom has to keep working.
SHELL_WINS = frozenset({"CUDA_VISIBLE_DEVICES"})


def load_dotenv(path: Path = ENV_FILE, override: bool = True) -> dict[str, str]:
    """Parse a `.env` file into os.environ.

    Supports `KEY=value`, `#` comments, and `${OTHER}` interpolation against
    values defined earlier in the same file or already in the environment.
    Deliberately dependency-free so `pip install python-dotenv` is not needed.

    With ``override=True`` (the default) the file wins over pre-existing
    environment variables, except for the keys in :data:`SHELL_WINS`.
    """
    parsed: dict[str, str] = {}
    if not path.is_file():
        return parsed

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        # Expand ${VAR} against this file first, then the live environment.
        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            return parsed.get(name, os.environ.get(name, ""))

        value = _VAR.sub(_sub, value)
        parsed[key] = value
        shell_set = key in SHELL_WINS and key in os.environ
        if (override or key not in os.environ) and not shell_set:
            os.environ[key] = value
    return parsed


def _path(key: str, default: str) -> Path:
    return Path(os.environ.get(key) or default).expanduser()


def _require_abs(name: str, p: Path) -> Path:
    if not p.is_absolute():
        raise ValueError(
            f"{name} must be an absolute path, got {p!r}. Fix it in {ENV_FILE}."
        )
    return p


@dataclass(frozen=True)
class Config:
    """Resolved configuration for one machine."""

    repo_root: Path
    raw_dataset: Path
    work: Path
    nnunet_raw: Path
    nnunet_preprocessed: Path
    nnunet_results: Path
    hff_data: Path
    hff_results: Path
    predictions: Path
    dataset_id: int
    dataset_name: str
    gamma: str
    nnunet_trainer: str
    swin_trainer: str
    swin_pretrained: str
    matlab: str
    extra: dict[str, str] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def training_dir(self) -> Path:
        return self.raw_dataset / "Training"

    @property
    def validation_dir(self) -> Path:
        return self.raw_dataset / "Validation"

    @property
    def dataset_dir(self) -> Path:
        """`$nnUNet_raw/Dataset501_BraTSPED`."""
        return self.nnunet_raw / self.dataset_name

    def member_prediction_dir(self, member: str) -> Path:
        """Where one ensemble member writes its probability maps."""
        return self.predictions / member

    def ensure_dirs(self) -> None:
        """Create every writable working directory."""
        for p in (
            self.work,
            self.nnunet_raw,
            self.nnunet_preprocessed,
            self.nnunet_results,
            self.hff_data,
            self.hff_results,
            self.predictions,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        lines = [
            "BraTS-PED configuration",
            f"  repo_root            {self.repo_root}",
            f"  raw_dataset          {self.raw_dataset}",
            f"  work                 {self.work}",
            f"  nnUNet_raw           {self.nnunet_raw}",
            f"  nnUNet_preprocessed  {self.nnunet_preprocessed}",
            f"  nnUNet_results       {self.nnunet_results}",
            f"  hff_data             {self.hff_data}",
            f"  hff_results          {self.hff_results}",
            f"  predictions          {self.predictions}",
            f"  dataset              {self.dataset_name} (id {self.dataset_id})",
            f"  gamma                {self.gamma or '(nnU-Net default init)'}",
            f"  nnunet_trainer       {self.nnunet_trainer}",
            f"  swin_trainer         {self.swin_trainer}",
            f"  swin_pretrained      {self.swin_pretrained or '(random init)'}",
        ]
        return "\n".join(lines)


def load_config() -> Config:
    load_dotenv()

    work = _path("PEDS_WORK", str(REPO_ROOT / "peds_work"))
    cfg = Config(
        repo_root=REPO_ROOT,
        raw_dataset=_path("PEDS_RAW_DATASET", str(REPO_ROOT / "raw_dataset" / "BraTS-PEDs-v1")),
        work=work,
        nnunet_raw=_path("nnUNet_raw", str(work / "nnUNet_raw")),
        nnunet_preprocessed=_path("nnUNet_preprocessed", str(work / "nnUNet_preprocessed")),
        nnunet_results=_path("nnUNet_results", str(work / "nnUNet_results")),
        hff_data=_path("PEDS_HFF_DATA", str(work / "hff_data")),
        hff_results=_path("PEDS_HFF_RESULTS", str(work / "hff_results")),
        predictions=_path("PEDS_PREDICTIONS", str(work / "predictions")),
        dataset_id=int(os.environ.get("PEDS_DATASET_ID", "501")),
        dataset_name=os.environ.get("PEDS_DATASET_NAME", "Dataset501_BraTSPED"),
        gamma=os.environ.get("PEDS_GAMMA", "0.7"),
        nnunet_trainer=os.environ.get("PEDS_NNUNET_TRAINER", "nnUNetTrainer_1000epochs"),
        swin_trainer=os.environ.get("PEDS_SWIN_TRAINER", "nnUNetTrainer_Swinunetr_1005epochs"),
        swin_pretrained=os.environ.get("PEDS_SWIN_PRETRAINED", ""),
        matlab=os.environ.get("PEDS_MATLAB", "matlab"),
    )

    for name, p in (
        ("PEDS_RAW_DATASET", cfg.raw_dataset),
        ("PEDS_WORK", cfg.work),
        ("nnUNet_raw", cfg.nnunet_raw),
        ("nnUNet_preprocessed", cfg.nnunet_preprocessed),
        ("nnUNet_results", cfg.nnunet_results),
    ):
        _require_abs(name, p)

    # nnU-Net reads these from the environment, including in subprocesses.
    os.environ["nnUNet_raw"] = str(cfg.nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(cfg.nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(cfg.nnunet_results)

    return cfg


CFG = load_config()


if __name__ == "__main__":
    print(CFG.describe())
