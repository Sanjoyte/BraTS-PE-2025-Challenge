"""The BraTS-PED label and region scheme — defined once, used everywhere.

The dataset conversion writes these regions into `dataset.json`, and the HFF
predictor has to decode its own outputs onto exactly the same axes. Keeping two
copies of the ordering in sync by hand is a silent-corruption risk, so both
import from here.

Voxel labels in the raw data:

===== ==== ==========================
Value Name Structure
===== ==== ==========================
1     ET   enhancing tumour
2     NET  non-enhancing tumour
3     CC   cystic component
4     ED   peritumoral oedema
===== ==== ==========================

BraTS scores *overlapping* structures, so the network is trained on regions
rather than the four disjoint labels.

nnU-Net rebuilds a label map by walking the regions in declaration order and
writing ``REGIONS_CLASS_ORDER[i]`` wherever region *i* fires, so later entries
overwrite earlier ones. Ordering coarse -> fine therefore reconstructs the
native label space exactly, and predictions come out directly submittable:

* ``WT`` covers the whole tumour       -> paint ED (4)
* ``TC`` covers the core               -> paint NET (2), the core default
* ``ED``/``NET``/``CC``                -> their own values
* ``ET`` is most specific, painted last -> 1
"""

from __future__ import annotations

# Region name -> the raw voxel labels it is the union of.
# INSERTION ORDER IS PART OF THE CONTRACT: coarse first, most specific last.
REGIONS: dict[str, list[int]] = {
    "WT": [1, 2, 3, 4],
    "TC": [1, 2, 3],
    "ED": [4],
    "NET": [2],
    "CC": [3],
    "ET": [1],
}

# The label each region paints, in the same order as REGIONS.
REGIONS_CLASS_ORDER: list[int] = [4, 2, 4, 2, 3, 1]

# The `labels` block for nnU-Net's dataset.json.
DATASET_JSON_LABELS: dict[str, object] = {"background": 0, **REGIONS}

# Raw label values, excluding background.
FOREGROUND_LABELS: list[int] = [1, 2, 3, 4]

CHANNEL_NAMES: dict[str, str] = {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"}

# Raw modality suffix -> nnU-Net channel index. Shared by every consumer so the
# channel meaning can never drift between conversion and inference.
CHANNEL_ORDER: dict[str, int] = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}

MODALITIES: list[str] = list(CHANNEL_ORDER)
HIGH_FREQ_BANDS: list[str] = ["H1", "H2", "H3", "H4"]

assert len(REGIONS) == len(REGIONS_CLASS_ORDER), (
    "REGIONS and REGIONS_CLASS_ORDER must stay the same length"
)


def decode_regions(region_mask_stack, out=None):
    """Turn a (n_regions, ...) boolean/probability stack into a label map.

    Mirrors nnU-Net's ``LabelManager`` region handling: walk the regions in
    order, painting ``REGIONS_CLASS_ORDER[i]`` wherever region *i* is above 0.5.
    """
    import numpy as np

    stack = np.asarray(region_mask_stack)
    if out is None:
        out = np.zeros(stack.shape[1:], dtype=np.uint8)
    for i, cls in enumerate(REGIONS_CLASS_ORDER):
        out[stack[i] > 0.5] = cls
    return out
