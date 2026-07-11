"""Class alignment across PlantVillage (lab, source) and two field targets:
PlantDoc and PlantWild. Unified label space = crop-disease pairs shared between
PlantVillage and PlantDoc (MSUN shared-subset protocol); PlantWild is mapped into
the SAME label space so source prototypes transfer.

All ~28 matchable PlantVillage<->PlantDoc pairs are listed. PlantWild covers 27 of
them (no `tomato_spider_mites` class). Empty target string = "not present in that
dataset" (silently skipped, not warned). Folder names differ across mirrors, so
`resolve_dirs` matches on a normalized name and warns about anything it can't find.
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
    pv: str             # PlantVillage folder (color split)
    plantdoc: str       # PlantDoc (Cropped) folder ("" = not present)
    plantwild: str      # PlantWild images/ folder ("" = not present)
    text: str           # CLIP zero-shot prompt


# key, PlantVillage, PlantDoc, PlantWild, CLIP prompt
_SPECS = [
    ClassSpec("apple_scab", "Apple___Apple_scab", "Apple Scab Leaf", "apple scab", "an apple leaf with apple scab"),
    ClassSpec("apple_rust", "Apple___Cedar_apple_rust", "Apple rust leaf", "apple rust", "an apple leaf with cedar apple rust"),
    ClassSpec("apple_healthy", "Apple___healthy", "Apple leaf", "apple leaf", "a healthy apple leaf"),
    ClassSpec("bellpepper_healthy", "Pepper,_bell___healthy", "Bell_pepper leaf", "bell pepper leaf", "a healthy bell pepper leaf"),
    ClassSpec("bellpepper_spot", "Pepper,_bell___Bacterial_spot", "Bell_pepper leaf spot", "bell pepper leaf spot", "a bell pepper leaf with bacterial leaf spot"),
    ClassSpec("blueberry_healthy", "Blueberry___healthy", "Blueberry leaf", "blueberry leaf", "a healthy blueberry leaf"),
    ClassSpec("cherry_healthy", "Cherry_(including_sour)___healthy", "Cherry leaf", "cherry leaf", "a healthy cherry leaf"),
    ClassSpec("corn_gray_leaf_spot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot", "corn gray leaf spot", "a corn leaf with gray leaf spot"),
    ClassSpec("corn_leaf_blight", "Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight", "corn northern leaf blight", "a corn leaf with northern leaf blight"),
    ClassSpec("corn_rust", "Corn_(maize)___Common_rust_", "Corn rust leaf", "corn rust", "a corn leaf with common rust"),
    ClassSpec("grape_healthy", "Grape___healthy", "grape leaf", "grape leaf", "a healthy grape leaf"),
    ClassSpec("grape_black_rot", "Grape___Black_rot", "grape leaf black rot", "grape black rot", "a grape leaf with black rot"),
    ClassSpec("peach_healthy", "Peach___healthy", "Peach leaf", "peach leaf", "a healthy peach leaf"),
    ClassSpec("potato_early_blight", "Potato___Early_blight", "Potato leaf early blight", "potato early blight", "a potato leaf with early blight"),
    ClassSpec("potato_late_blight", "Potato___Late_blight", "Potato leaf late blight", "potato late blight", "a potato leaf with late blight"),
    ClassSpec("raspberry_healthy", "Raspberry___healthy", "Raspberry leaf", "raspberry leaf", "a healthy raspberry leaf"),
    ClassSpec("soybean_healthy", "Soybean___healthy", "Soyabean leaf", "soybean leaf", "a healthy soybean leaf"),
    ClassSpec("squash_powdery_mildew", "Squash___Powdery_mildew", "Squash Powdery mildew leaf", "squash powdery mildew", "a squash leaf with powdery mildew"),
    ClassSpec("strawberry_healthy", "Strawberry___healthy", "Strawberry leaf", "strawberry leaf", "a healthy strawberry leaf"),
    ClassSpec("tomato_early_blight", "Tomato___Early_blight", "Tomato Early blight leaf", "tomato early blight", "a tomato leaf with early blight"),
    ClassSpec("tomato_healthy", "Tomato___healthy", "Tomato leaf", "tomato leaf", "a healthy tomato leaf"),
    ClassSpec("tomato_bacterial_spot", "Tomato___Bacterial_spot", "Tomato leaf bacterial spot", "tomato bacterial leaf spot", "a tomato leaf with bacterial spot"),
    ClassSpec("tomato_late_blight", "Tomato___Late_blight", "Tomato leaf late blight", "tomato late blight", "a tomato leaf with late blight"),
    ClassSpec("tomato_mosaic_virus", "Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus", "tomato mosaic virus", "a tomato leaf with mosaic virus"),
    ClassSpec("tomato_yellow_curl_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus", "tomato yellow leaf curl virus", "a tomato leaf with yellow leaf curl virus"),
    ClassSpec("tomato_leaf_mold", "Tomato___Leaf_Mold", "Tomato mold leaf", "tomato leaf mold", "a tomato leaf with leaf mold"),
    ClassSpec("tomato_septoria", "Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot", "tomato septoria leaf spot", "a tomato leaf with septoria leaf spot"),
    ClassSpec("tomato_spider_mites", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato two spotted spider mites leaf", "", "a tomato leaf damaged by two-spotted spider mites"),
]

SHARED_CLASSES = [s.key for s in _SPECS]
CLASS_TO_IDX = {key: i for i, key in enumerate(SHARED_CLASSES)}
IDX_TO_CLASS = {i: key for key, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(SHARED_CLASSES)
_SPEC_BY_KEY = {s.key: s for s in _SPECS}


def clip_prompts() -> list[str]:
    return [_SPEC_BY_KEY[IDX_TO_CLASS[i]].text for i in range(NUM_CLASSES)]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _expected_name(spec: ClassSpec, which: str) -> str:
    return {"pv": spec.pv, "plantdoc": spec.plantdoc, "plantwild": spec.plantwild}[which]


def resolve_dirs(root: str, which: str, split: str = "all") -> dict[str, int]:
    """Map each *actual* class directory under `root` to a unified label index.

    `which` in {'pv','plantdoc','plantwild'}. `split` (PlantDoc only) is 'train',
    'test', or 'all' (combines both). Every matching directory is included. Specs
    with an empty name for this dataset are skipped silently; missing expected
    classes are warned.
    """
    assert which in ("pv", "plantdoc", "plantwild")
    assert split in ("train", "test", "all")

    if which == "plantdoc":
        wanted = {"train": ("train", "TRAIN"), "test": ("test", "TEST"),
                  "all": ("train", "TRAIN", "test", "TEST")}[split]
        search_roots = [os.path.join(root, s) for s in wanted
                        if os.path.isdir(os.path.join(root, s))]
        if not search_roots:
            search_roots = [root]
    else:  # pv, plantwild: class folders sit directly under root
        search_roots = [root]

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
        expected = _expected_name(spec, which)
        if not expected:            # class not present in this dataset — skip, don't warn
            continue
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
