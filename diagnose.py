
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import bus_seg as B                      # باید کنار bus_seg.py باشد (SPLIT_MODE = "baseline")

CKPT = r"C:\Users\amir\DL\runs\baseline_rerun\best.pth"   # baseline؛ یا مسیر یک best_full.pth
SPLIT = "val"                                        # "val" (برای تصمیم) یا "train" (برای دیدن overfit)
OUT_DIR = B.RUNS_DIR / "diagnose_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

names = sorted(p.name for p in (B.ROOT / "images").glob("*.png"))
info = pd.read_csv(B.ROOT / "INFO.csv", sep=";")
info["Image"] = info["Image"].str.strip()
info = info.set_index("Image")
groups = [n.split("_")[0] for n in names]
tr, va, _test_unused = B.baseline_split(names, groups)        # test عمداً استفاده نمی‌شود
idx = {"train": tr, "val": va}[SPLIT]

model = B.MBV3SmallUNet(B.NUM_CLASSES, B.WEIGHTS).to(B.DEVICE)
ck = torch.load(CKPT, map_location=B.DEVICE, weights_only=False)
model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
model.eval()

yn = lambda s: "Yes" if str(s).strip().lower().startswith("y") else "No"
rows, store = [], {}
for i in idx:
    n = names[i]
    x_u8, y = B.load_pair(n)
    xf = x_u8.float() / 255.0
    gray = xf.mean(0)
    with torch.no_grad():
        lg = model(((xf - B.MEAN) / B.STD)[None].to(B.DEVICE))
    pred, gt = (lg.argmax(1)[0] > 0).cpu(), y > 0
    inter, ps, gs = (pred & gt).sum().item(), pred.sum().item(), gt.sum().item()
    q = torch.quantile(gray.flatten(), torch.tensor([0.01, 0.99]))
    r = dict(name=n, patient=n.split("_")[0], Label=info.loc[n, "Label"].strip(),
             Marks=yn(info.loc[n, "Marks"]), Doppler=yn(info.loc[n, "Doppler"]),
             mean=gray.mean().item(), std=gray.std().item(), dyn_range=(q[1] - q[0]).item(),
             black_frac=(gray < 0.04).float().mean().item(),
             color=(xf.max(0).values - xf.min(0).values).mean().item(),   # ~0 برای تصویر خاکستری
             gs=gs, ps=ps, area_frac=gt.float().mean().item(),
             dice=2 * inter / (ps + gs) if gs > 0 else np.nan,
             fp=float(ps > B.MIN_PIXELS) if gs == 0 else np.nan, contrast=np.nan, edge=np.nan)
    if gs > 0:
        ring = (F.max_pool2d(gt[None, None].float(), 41, 1, 20)[0, 0] > 0) & ~gt
        if ring.any():
            r["contrast"] = (gray[gt].mean() - gray[ring].mean()).item()     # روشنایی ضایعه − اطرافش
        r["edge"] = float(gt[0].any() or gt[-1].any() or gt[:, 0].any() or gt[:, -1].any())
        store[n] = (x_u8, gt, pred)
    rows.append(r)

d = pd.DataFrame(rows)
d.to_csv(OUT_DIR / "per_image.csv", index=False)
les, nor = d[d.gs > 0].copy(), d[d.gs == 0]
print(f"\n[{SPLIT}] lesion images={len(les)} ({les.patient.nunique()} patients) | normal={len(nor)}")
print(f"mean dice (lesion, per-image) = {les.dice.mean():.3f} | fp_normal = {nor.fp.mean():.2f}\n")

num = ["mean", "std", "dyn_range", "black_frac", "color", "area_frac", "contrast", "edge"]
print("Spearman(dice, feature) روی تصاویر دارای ضایعه:")
print(les[num + ["dice"]].corr(method="spearman")["dice"].drop("dice").round(2).to_string(), "\n")
for c in ("Label", "Marks", "Doppler"):
    print(les.groupby(c).dice.agg(["count", "mean"]).round(3).to_string(), "\n")
for c in ("area_frac", "contrast"):
    les[c + "_bin"] = pd.qcut(les[c], 3, labels=["low", "mid", "high"], duplicates="drop")
    print(les.groupby(c + "_bin", observed=True).dice.agg(["count", "mean"]).round(3).to_string(), "\n")
print("false-positive روی تصاویر normal (بر حسب Marks):")
print(nor.groupby("Marks").fp.agg(["count", "mean"]).round(2).to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1) ۸ تصویر بدترین dice: عکس / GT / پیش‌بینی
worst = les.sort_values("dice").head(8).name.tolist()
fig, ax = plt.subplots(3, len(worst), figsize=(3 * len(worst), 7))
for j, n in enumerate(worst):
    x_u8, gt, pred = store[n]
    ax[0, j].imshow(x_u8.permute(1, 2, 0).numpy()); ax[0, j].set_title(f"{n[:-4]}\n{d.set_index('name').loc[n, 'dice']:.2f}", fontsize=8)
    ax[1, j].imshow(gt); ax[2, j].imshow(pred)
    for k in range(3): ax[k, j].axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "worst8.png", dpi=90); plt.close()

# 2) ۲۴ تصویر تصادفی از train+val برای دیدن تنوع خام (test شامل نیست)
rng = random.Random(0)
pick = rng.sample(tr + va, 24)
fig, ax = plt.subplots(4, 6, figsize=(18, 9))
for a, i in zip(ax.flat, pick):
    n = names[i]; x_u8, _ = B.load_pair(n)
    a.imshow(x_u8.permute(1, 2, 0).numpy()); a.axis("off")
    a.set_title(f"{n[:-4]} {info.loc[n, 'Label'].strip()[:3]} M:{yn(info.loc[n, 'Marks'])[0]} D:{yn(info.loc[n, 'Doppler'])[0]}", fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR / "random24.png", dpi=80); plt.close()
print("\nsaved in", OUT_DIR)