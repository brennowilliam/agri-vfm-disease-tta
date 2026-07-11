# Paper 1 — Code (Phase 0–1 scaffold)

Reproduces the lab→field collapse baseline and builds the frozen-feature harness for
VFM test-time adaptation. Designed to run on **Kaggle free tier** (2×T4) or Colab — no local GPU.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Class alignment (PlantVillage ↔ PlantDoc, MSUN shared subset) | `class_alignment.py` | the unified label space, imported everywhere |
| Data loaders | `datasets.py` | aligned `Dataset`s for PV / PlantDoc / PlantWild / Cassava |
| **Phase 1** — source training + collapse | `train_source.py` | source checkpoint + source-only PlantDoc accuracy (~30.78% target) |
| **Phase 2** — frozen features | `extract_features.py` | cached DINOv2 / CLIP embeddings (`.npy`) for every dataset |
| Baselines from cache | `eval_baselines.py` | DINOv2 linear-probe / kNN, CLIP zero-shot |
| **Phase 3** — the method | `tta.py` | source-anchored, imbalance-gated online prototype TTA (backprop-free, CPU) |

## Quickstart (Kaggle)

1. Add these datasets to a Kaggle Notebook (Add Data):
   - PlantVillage — `abdallahalidev/plantvillage-dataset` (use the `color/` split)
   - PlantDoc — upload the `pratikkayal/PlantDoc-Dataset` repo (Cropped-PlantDoc) as a dataset
   - PlantWild — `uqtwei2/PlantWild` (HF; mirror to a Kaggle dataset)
   - Cassava — `cassava-leaf-disease-classification` (competition data)
2. `pip install -q timm open_clip_torch`
3. Set paths (env vars or edit `config.py`), then:
   ```bash
   python train_source.py --backbone resnet50 --epochs 15      # Phase 1: reproduce collapse
   python extract_features.py --backbone dinov2_vitb14          # Phase 2: cache features
   python eval_baselines.py --backbone dinov2_vitb14            # linear-probe / kNN
   ```

## Configuration

`config.py` reads dataset roots from env vars (`PV_ROOT`, `PLANTDOC_ROOT`, `PLANTWILD_ROOT`,
`CASSAVA_ROOT`) and falls back to Kaggle `/kaggle/input/...` defaults. Override to run locally.

## ⚠️ Verify before trusting exact numbers

- **The shared class subset in `class_alignment.py` lists all ~28 matchable PlantVillage↔PlantDoc
  crop-disease pairs. MSUN reports 20.** To match their exact 30.78% source-only number, pare
  `SHARED_CLASSES` down to MSUN's specific 20 (confirm from the paper). The *collapse itself*
  (~99% lab → ~30% field) reproduces regardless of the exact subset.
- Folder names vary across dataset mirrors; the alignment resolver normalizes and warns on any
  expected class it can't find — check its warnings on first run.
