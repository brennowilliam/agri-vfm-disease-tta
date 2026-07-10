"""Phase 1 — train the source model on PlantVillage (aligned subset) and measure the
lab->field collapse on Cropped-PlantDoc.

Expected: ~95-99% on a PlantVillage held-out val split, collapsing to ~30% on PlantDoc
(source-only, zero adaptation). That collapse is the baseline the whole paper attacks.

    python train_source.py --backbone resnet50 --epochs 15
"""
from __future__ import annotations

import argparse
import os

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
from config import CFG, BACKBONES
import datasets as D
import metrics as M


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        logits = model(imgs.to(device))
        ps.append(logits.argmax(1).cpu())
        ys.append(labels)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_b16"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=CFG.batch_size)
    args = ap.parse_args()

    D.seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, timm_name = BACKBONES[args.backbone]
    print(f"Backbone: {args.backbone} ({timm_name}) | classes: {NUM_CLASSES} | device: {device}")

    # ---- Data: PlantVillage source (train/val), PlantDoc target (eval) ----
    pv_train_ds = D.plantvillage(transform=D.build_transforms(train=True))
    pv_eval_ds = D.plantvillage(transform=D.build_transforms(train=False))
    tr_idx, va_idx = D.stratified_split(pv_train_ds)
    train_loader = DataLoader(Subset(pv_train_ds, tr_idx), batch_size=args.batch,
                              shuffle=True, num_workers=CFG.num_workers, drop_last=True)
    val_loader = DataLoader(Subset(pv_eval_ds, va_idx), batch_size=args.batch,
                            shuffle=False, num_workers=CFG.num_workers)
    pd_loader = DataLoader(D.plantdoc(transform=D.build_transforms(train=False)),
                           batch_size=args.batch, shuffle=False, num_workers=CFG.num_workers)

    # ---- Model ----
    model = timm.create_model(timm_name, pretrained=True, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    # ---- Train ----
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss = crit(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()
        yv, pv = predict(model, val_loader, device)
        print(f"epoch {epoch+1}: loss={running/len(train_loader):.3f} "
              f"val_acc={M.accuracy(yv, pv)*100:.2f}%")

    # ---- The headline: source-only lab vs field ----
    val_s = M.summarize(*predict(model, val_loader, device), NUM_CLASSES, IDX_TO_CLASS)
    M.print_summary("PlantVillage val (in-domain, lab)", val_s)
    pd_s = M.summarize(*predict(model, pd_loader, device), NUM_CLASSES, IDX_TO_CLASS)
    M.print_summary("PlantDoc (source-only transfer, FIELD)", pd_s)
    print(f"\n>>> COLLAPSE: {val_s['accuracy']*100:.1f}% (lab) -> {pd_s['accuracy']*100:.1f}% (field). "
          f"Target from MSUN: ~99% -> ~30.78%.")

    os.makedirs(CFG.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(CFG.ckpt_dir, f"source_{args.backbone}.pt")
    torch.save({"model": model.state_dict(), "backbone": args.backbone,
                "timm_name": timm_name, "num_classes": NUM_CLASSES,
                "idx_to_class": IDX_TO_CLASS}, ckpt)
    print(f"Saved source checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
