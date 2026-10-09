
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

# =====================================================================
# 1) CONFIG
# =====================================================================
ROOT = Path(r"C:\Users\amir\DL\datasets\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM")
WEIGHTS = Path(r"C:\Users\amir\DL\models\mobilenet_v3_small\mobilenet_v3_small.pth")
OUT_DIR = Path(r"C:\Users\amir\DL\runs\bus_uclm")

NUM_CLASSES = 2          # 2 = ضایعه/پس‌زمینه ،  3 = پس‌زمینه/خوش‌خیم(سبز)/بدخیم(قرمز)
IMG_W, IMG_H = 384, 256
BATCH_SIZE = 8
EPOCHS = 40
SEED = 42
CROP_ARTIFACTS = False   # True = کراپ حاشیه (نشانگرهای گوشه)

# --- فریز و learning rate ---
FROZEN_STAGES = ["s2", "s4", "s8"]   # کاملاً فریز (لایه‌های اول)
ENC_LR = 3e-4                        # max LR برای s16 و s32 (encoder قابل‌آموزش)
DEC_LR = 1e-3                        # max LR برای decoder
WARMUP_FRAC = 0.2                    # ۲۰٪ اول: LR از کم به زیاد؛ بعدش کاهش تدریجی
WEIGHT_DECAY = 1e-4

# --- پایش گرادیان ---
GRAD_REPORT_EVERY = 10               # هر چند epoch گزارش چاپ شود
GRAD_LOW_REL = 0.01                  # اگر نسبت گرادیان یک ماژول < 1٪ بهترین ماژول باشد => LOW

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


set_seed(SEED)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# =====================================================================
# 2) DATA
# =====================================================================
def color_mask_to_class(rgb: np.ndarray, num_classes: int) -> np.ndarray:
    """سبز=خوش‌خیم، قرمز=بدخیم، سیاه=پس‌زمینه."""
    r, g = rgb[..., 0].astype(int), rgb[..., 1].astype(int)
    fg = rgb.max(-1) > 60
    out = np.zeros(rgb.shape[:2], np.uint8)
    if num_classes == 2:
        out[fg] = 1
    else:
        out[fg & (g >= r)] = 1
        out[fg & (r > g)] = 2
    return out


def load_pair(name: str):
    img = Image.open(ROOT / "images" / name).convert("RGB")
    msk = Image.open(ROOT / "masks" / name).convert("RGB")
    if CROP_ARTIFACTS:
        w, h = img.size
        box = (0, int(0.04 * h), w, int(0.90 * h))
        img, msk = img.crop(box), msk.crop(box)
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    msk = msk.resize((IMG_W, IMG_H), Image.NEAREST)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    y = torch.from_numpy(color_mask_to_class(np.array(msk), NUM_CLASSES)).long()
    return x, y


def find_groups(df, names):
    """پیشوند اسم فایل (ALWI, ANAT, ...) = شناسه بیمار."""
    g = [n.split("_")[0] for n in names]
    k = len(set(g))
    print(f"[split] groups from filename prefix: {k} groups")
    if k < 10:
        print("[split] WARNING: تعداد گروه کم است، split تصادفی استفاده می‌شود")
        return None
    return g


class BUSDataset(Dataset):
    def __init__(self, names, cache, train):
        self.names, self.cache, self.train = names, cache, train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        x, y = self.cache[self.names[i]]
        x = x.float() / 255.0
        if self.train:
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.8:
                x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1)).clamp(0, 1)
            if random.random() < 0.5:
                s = random.uniform(0.85, 1.0)
                h, w = int(IMG_H * s), int(IMG_W * s)
                t, l = random.randint(0, IMG_H - h), random.randint(0, IMG_W - w)
                x = F.interpolate(x[None, :, t:t + h, l:l + w], (IMG_H, IMG_W), mode="bilinear", align_corners=False)[0]
                y = F.interpolate(y[None, None, t:t + h, l:l + w].float(), (IMG_H, IMG_W), mode="nearest")[0, 0].long()
        x = (x - MEAN) / STD
        return x, y


def build_loaders():
    names = sorted(p.name for p in (ROOT / "images").glob("*.png"))
    assert names == sorted(p.name for p in (ROOT / "masks").glob("*.png")), "images/masks mismatch"
    print(f"{len(names)} pairs found, loading into RAM ...")
    cache = {n: load_pair(n) for n in names}

    df = pd.read_csv(ROOT / "INFO.csv", sep=";")
    print(df["Label"].value_counts())
    groups = find_groups(df, names)
    idx = np.arange(len(names))
    if groups is not None:
        g = np.array(groups)
        tr, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(idx, groups=g))
        va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(rest, groups=g[rest]))
        va, te = rest[va_rel], rest[te_rel]
    else:
        tr, rest = train_test_split(idx, test_size=0.3, random_state=SEED)
        va, te = train_test_split(rest, test_size=0.5, random_state=SEED)

    mk = lambda ids, train: BUSDataset([names[i] for i in ids], cache, train)
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    return (DataLoader(mk(tr, True), BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True),
            DataLoader(mk(va, False), BATCH_SIZE, num_workers=0),
            DataLoader(mk(te, False), BATCH_SIZE, num_workers=0))


# =====================================================================
# 3) MODEL  (با فریز کامل لایه‌های اول)
# =====================================================================
def conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.block = nn.Sequential(conv_bn(in_c + skip_c, out_c), conv_bn(out_c, out_c))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], 1))


class MBV3SmallUNet(nn.Module):
    ENCODER_STAGES = ["s2", "s4", "s8", "s16", "s32"]

    def __init__(self, num_classes, weights_path=None, frozen=()):
        super().__init__()
        net = mobilenet_v3_small(weights=None)
        if weights_path and Path(weights_path).exists():
            obj = torch.load(weights_path, map_location="cpu", weights_only=False)
            sd = obj.state_dict() if isinstance(obj, nn.Module) else obj
            print("load pretrained:", net.load_state_dict(sd, strict=False))
        else:
            print("WARNING: pretrained weights not found, training from scratch")
        f = net.features
        self.s2, self.s4, self.s8 = f[0:1], f[1:2], f[2:4]    # 16, 16, 24 ch
        self.s16, self.s32 = f[4:9], f[9:13]                  # 48, 576 ch
        self.u1 = Up(576, 48, 128)
        self.u2 = Up(128, 24, 64)
        self.u3 = Up(64, 16, 32)
        self.u4 = Up(32, 16, 24)
        self.head = nn.Conv2d(24, num_classes, 1)

        self.frozen = list(frozen)
        for n in self.frozen:
            for p in getattr(self, n).parameters():
                p.requires_grad = False

    def train(self, mode=True):
        """لایه‌های فریز در eval می‌مانند تا آمار BatchNorm آن‌ها تغییر نکند."""
        super().train(mode)
        for n in self.frozen:
            getattr(self, n).eval()
        return self

    def forward(self, x):
        size = x.shape[-2:]
        c2 = self.s2(x); c4 = self.s4(c2); c8 = self.s8(c4)
        c16 = self.s16(c8); c32 = self.s32(c16)
        d = self.u1(c32, c16); d = self.u2(d, c8); d = self.u3(d, c4); d = self.u4(d, c2)
        return F.interpolate(self.head(d), size=size, mode="bilinear", align_corners=False)

    def param_groups(self):
        """(encoder قابل‌آموزش, decoder) برای دو LR متفاوت."""
        enc = [p for n in self.ENCODER_STAGES if n not in self.frozen for p in getattr(self, n).parameters()]
        dec = [p for n in ("u1", "u2", "u3", "u4", "head") for p in getattr(self, n).parameters()]
        return enc, dec

    def trainable_modules(self):
        return [n for n in self.ENCODER_STAGES if n not in self.frozen] + ["u1", "u2", "u3", "u4", "head"]


# =====================================================================
# 4) LOSS + METRICS
# =====================================================================
def dice_loss(logits, y, eps=1.0):
    p = logits.softmax(1)
    oh = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps))[1:].mean()


def loss_fn(logits, y):
    return F.cross_entropy(logits, y) + dice_loss(logits, y)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    C = NUM_CLASSES
    inter, pred_s, gt_s = torch.zeros(C), torch.zeros(C), torch.zeros(C)
    tot_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        lg = model(x)
        tot_loss += loss_fn(lg, y).item() * len(x); n += len(x)
        p = lg.argmax(1)
        for c in range(1, C):
            inter[c] += ((p == c) & (y == c)).sum().item()
            pred_s[c] += (p == c).sum().item()
            gt_s[c] += (y == c).sum().item()
    dice = 2 * inter[1:] / (pred_s[1:] + gt_s[1:]).clamp(min=1)
    iou = inter[1:] / (pred_s[1:] + gt_s[1:] - inter[1:]).clamp(min=1)
    return tot_loss / n, dice.mean().item(), iou.mean().item()


# =====================================================================
# 5) GRADIENT MONITOR
# =====================================================================
def _norm(params, use_grad):
    ts = [(p.grad if use_grad else p.data).float().norm() ** 2 for p in params if (p.grad is not None or not use_grad)]
    return torch.stack(ts).sum().sqrt().item() if ts else 0.0


class GradMonitor:
    """میانگین نُرم گرادیان هر ماژول در طول یک epoch را جمع می‌کند."""

    def __init__(self, model):
        self.model, self.mods = model, model.trainable_modules()
        self.reset()

    def reset(self):
        self.acc, self.n = {m: 0.0 for m in self.mods}, 0

    def update(self):  # بعد از backward و قبل از step صدا زده شود
        for m in self.mods:
            self.acc[m] += _norm(getattr(self.model, m).parameters(), use_grad=True)
        self.n += 1

    def report(self, ep):
        avg = {m: self.acc[m] / max(self.n, 1) for m in self.mods}
        rel = {m: avg[m] / max(_norm(getattr(self.model, m).parameters(), use_grad=False), 1e-12) for m in self.mods}
        best = max(rel.values()) or 1e-12
        print(f"\n--- GRADIENT REPORT (epoch {ep}) ---")
        print(f"{'module':<8}{'grad_norm':>12}{'grad/weight':>14}  status")
        rows = []
        for m in self.mods:
            if avg[m] < 1e-8:
                st = "DEAD (گرادیان صفر)"
            elif rel[m] < GRAD_LOW_REL * best:
                st = "LOW  (گرادیان خیلی کم)"
            else:
                st = "ok"
            print(f"{m:<8}{avg[m]:>12.3e}{rel[m]:>14.3e}  {st}")
            rows.append([ep, m, avg[m], rel[m], st])
        frozen_ok = all(p.grad is None or p.grad.abs().sum().item() == 0 for n in self.model.frozen
                        for p in getattr(self.model, n).parameters())
        print(f"frozen stages {self.model.frozen}: no gradients = {frozen_ok}\n")
        path = OUT_DIR / "grad_log.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "module", "grad_norm", "grad_over_weight", "status"])
            w.writerows(rows)


# =====================================================================
# 6) TRAIN + TEST + VISUALIZATION
# =====================================================================
def save_preview(model, loader, path, k=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    x, y = next(iter(loader))
    with torch.no_grad():
        p = model(x.to(DEVICE)).argmax(1).cpu()
    k = min(k, len(x))
    fig, ax = plt.subplots(3, k, figsize=(3 * k, 7))
    for i in range(k):
        img = (x[i] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        ax[0, i].imshow(img); ax[1, i].imshow(y[i], vmin=0, vmax=NUM_CLASSES - 1); ax[2, i].imshow(p[i], vmin=0, vmax=NUM_CLASSES - 1)
        for r in range(3):
            ax[r, i].axis("off")
    ax[0, 0].set_title("image"); ax[1, 0].set_title("GT"); ax[2, 0].set_title("pred")
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


def main():
    print("device:", DEVICE)
    tr_dl, va_dl, te_dl = build_loaders()
    model = MBV3SmallUNet(NUM_CLASSES, WEIGHTS, frozen=FROZEN_STAGES).to(DEVICE)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"frozen: {FROZEN_STAGES} ({n_fr:,} params) | trainable: {model.trainable_modules()} ({n_tr:,} params)")

    enc_params, dec_params = model.param_groups()
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": ENC_LR}, {"params": dec_params, "lr": DEC_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    # LR: کم -> زیاد -> کم  (هر batch یک step)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[ENC_LR, DEC_LR], total_steps=EPOCHS * len(tr_dl),
        pct_start=WARMUP_FRAC, anneal_strategy="cos", div_factor=20, final_div_factor=100,
    )
    gmon = GradMonitor(model)
    best, ckpt = -1.0, OUT_DIR / "best.pth"

    for ep in range(1, EPOCHS + 1):
        model.train(); run = 0.0; gmon.reset()
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            gmon.update()
            opt.step(); sched.step()
            run += loss.item()
        vl, vd, vi = evaluate(model, va_dl)
        flag = ""
        if vd > best:
            best = vd; torch.save(model.state_dict(), ckpt); flag = "  <- best"
        lr_e, lr_d = opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]
        print(f"ep {ep:02d}/{EPOCHS} lr(enc/dec) {lr_e:.1e}/{lr_d:.1e} train_loss {run / len(tr_dl):.4f} "
              f"| val_loss {vl:.4f} dice {vd:.4f} iou {vi:.4f}{flag}")
        if ep == 1 or ep % GRAD_REPORT_EVERY == 0:
            gmon.report(ep)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    tl, td, ti = evaluate(model, te_dl)
    print(f"\nTEST  dice {td:.4f}  iou {ti:.4f}")
    save_preview(model, te_dl, OUT_DIR / "test_preview.png")
    print("saved:", ckpt, "and", OUT_DIR / "test_preview.png", "and", OUT_DIR / "grad_log.csv")


if __name__ == "__main__":
    main()
