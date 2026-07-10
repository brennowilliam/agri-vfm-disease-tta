"""PlantVillage (lab, source) <-> PlantDoc (field, target) class alignment.

This defines the unified label space for the lab->field benchmark following the
MSUN "shared-subset" protocol (Wu et al., Plant Phenomics 2023): keep only the
crop-disease pairs present in BOTH datasets.

Below are ALL ~28 matchable pairs. MSUN reports 20 classes; to match their exact
30.78% source-only number, pare `SHARED_CLASSES` to their specific 20 (verify from
the paper). The collapse (~99% lab -> ~30% field) reproduces regardless of the subset.

Folder names differ across dataset mirrors, so `resolve_dirs()` matches on a
normalized name and warns about anything it can't find.
"""
from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassSpec:
    key: str            # canonical unified label
    pv: str             # PlantVillage folder name (color split)
    plantdoc: str       # PlantDoc (Cropped) folder name
    text: str           # CLIP zero-shot prompt (natural-language description)


# key, PlantVillage folder, PlantDoc folder, CLIP prompt
_SPECS = [
    ClassSpec("apple_scab", "Apple___Apple_scab", "Apple Scab Leaf", "an apple leaf with apple scab"),
    ClassSpec("apple_rust", "Apple___Cedar_apple_rust", "Apple rust leaf", "an apple leaf with cedar apple rust"),
    ClassSpec("apple_healthy", "Apple___healthy", "Apple leaf", "a healthy apple leaf"),
    ClassSpec("bellpepper_healthy", "Pepper,_bell___healthy", "Bell_pepper leaf", "a healthy bell pepper leaf"),
    ClassSpec("bellpepper_spot", "Pepper,_bell___Bacterial_spot", "Bell_pepper leaf spot", "a bell pepper leaf with bacterial leaf spot"),
    ClassSpec("blueberry_healthy", "Blueberry___healthy", "Blueberry leaf", "a healthy blueberry leaf"),
    ClassSpec("cherry_healthy", "Cherry_(including_sour)___healthy", "Cherry leaf", "a healthy cherry leaf"),
    ClassSpec("corn_gray_leaf_spot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot", "a corn leaf with gray leaf spot"),
    ClassSpec("corn_leaf_blight", "Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight", "a corn leaf with northern leaf blight"),
    ClassSpec("corn_rust", "Corn_(maize)___Common_rust_", "Corn rust leaf", "a corn leaf with common rust"),
    ClassSpec("grape_healthy", "Grape___healthy", "grape leaf", "a healthy grape leaf"),
    ClassSpec("grape_black_rot", "Grape___Black_rot", "grape leaf black rot", "a grape leaf with black rot"),
    ClassSpec("peach_healthy", "Peach___healthy", "Peach leaf", "a healthy peach leaf"),
    ClassSpec("potato_early_blight", "Potato___Early_blight", "Potato leaf early blight", "a potato leaf with early blight"),
    ClassSpec("potato_late_blight", "Potato___Late_blight", "Potato leaf late blight", "a potato leaf with late blight"),
    ClassSpec("raspberry_healthy", "Raspberry___healthy", "Raspberry leaf", "a healthy raspberry leaf"),
    ClassSpec("soybean_healthy", "Soybean___healthy", "Soyabean leaf", "a healthy soybean leaf"),
    ClassSpec("squash_powdery_mildew", "Squash___Powdery_mildew", "Squash Powdery mildew leaf", "a squash leaf with powdery mildew"),
    ClassSpec("strawberry_healthy", "Strawberry___healthy", "Strawberry leaf", "a healthy strawberry leaf"),
    ClassSpec("tomato_early_blight", "Tomato___Early_blight", "Tomato Early blight leaf", "a tomato leaf with early blight"),
    ClassSpec("tomato_healthy", "Tomato___healthy", "Tomato leaf", "a healthy tomato leaf"),
    ClassSpec("tomato_bacterial_spot", "Tomato___Bacterial_spot", "Tomato leaf bacterial spot", "a tomato leaf with bacterial spot"),
    ClassSpec("tomato_late_blight", "Tomato___Late_blight", "Tomato leaf late blight", "a tomato leaf with late blight"),
    ClassSpec("tomato_mosaic_virus", "Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus", "a tomato leaf with mosaic virus"),
    ClassSpec("tomato_yellow_curl_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus", "a tomato leaf with yellow leaf curl virus"),
    ClassSpec("tomato_leaf_mold", "Tomato___Leaf_Mold", "Tomato mold leaf", "a tomato leaf with leaf mold"),
    ClassSpec("tomato_septoria", "Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot", "a tomato leaf with septoria leaf spot"),
    ClassSpec("tomato_spider_mites", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato two spotted spider mites leaf", "a tomato leaf damaged by two-spotted spider mites"),
]

# The active shared subset. To reproduce MSUN's exact 20, replace with their subset of keys.
SHARED_CLASSES = [s.key for s in _SPECS]

CLASS_TO_IDX = {key: i for i, key in enumerate(SHARED_CLASSES)}
IDX_TO_CLASS = {i: key for key, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(SHARED_CLASSES)

_SPEC_BY_KEY = {s.key: s for s in _SPECS}


def clip_prompts() -> list[str]:
    """Natural-language prompt per class index (for CLIP zero-shot)."""
    return [_SPEC_BY_KEY[IDX_TO_CLASS[i]].text for i in range(NUM_CLASSES)]


def _norm(name: str) -> str:
    """Lowercase, strip punctuation/whitespace so mirror naming differences match."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_dirs(root: str, which: str, split: str = "all") -> dict[str, int]:
    """Map each *actual* class directory under `root` to a unified label index.

    `which` is 'pv' or 'plantdoc'. `split` (PlantDoc only) is 'train', 'test', or 'all'
    — 'all' combines both so evaluation covers the full field target. Every matching
    directory is included (a class present in both train/ and test/ contributes both).
    Directories not in the shared subset are skipped; warns for any missing expected class.
    """
    assert which in ("pv", "plantdoc")
    assert split in ("train", "test", "all")

    if which == "plantdoc":
        wanted = {"train": ("train", "TRAIN"), "test": ("test", "TEST"),
                  "all": ("train", "TRAIN", "test", "TEST")}[split]
        search_roots = [os.path.join(root, s) for s in wanted
                        if os.path.isdir(os.path.join(root, s))]
        if not search_roots:            # classes sit directly under root
            search_roots = [root]
    else:
        search_roots = [root]

    # normalized class name -> list of actual dir paths (keep ALL matches, e.g. train+test).
    available: dict[str, list[str]] = defaultdict(list)
    for sr in search_roots:
        if not os.path.isdir(sr):
            continue
        for d in sorted(os.listdir(sr)):
            full = os.path.join(sr, d)
            if os.path.isdir(full):
                available[_norm(d)].append(full)

    mapping: dict[str, int] = {}
    missing: list[str] = []
    for spec in _SPECS:
        if spec.key not in CLASS_TO_IDX:
            continue
        expected = spec.pv if which == "pv" else spec.plantdoc
        hits = available.get(_norm(expected), [])
        if not hits:
            missing.append(f"{spec.key} ({expected})")
        for h in hits:
            mapping[h] = CLASS_TO_IDX[spec.key]
    if missing:
        warnings.warn(
            f"[class_alignment] {which}: {len(missing)} expected class dir(s) not found under "
            f"{root}: {missing}. Check the dataset mirror's folder names.")
    return mapping
