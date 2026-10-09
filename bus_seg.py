
import json
import random
import time
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
# 1) CONFIG  (فقط این بخش را در صورت نیاز عوض کن)
# =====================================================================
ROOT = Path(r"C:\Users\amir\DL\datasets\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM")
WEIGHTS = Path(r"C:\Users\amir\DL\models\mobilenet_v3_small\mobilenet_v3_small.pth")

RUNS_DIR = Path(r"C:\Users\amir\DL\runs")          # [NEW] diagnose.py از این استفاده می‌کند
RUN_NAME = "baseline_rerun"                         # [NEW] برای هر آزمایش جدید اسم جدید بگذار
OUT_DIR = RUNS_DIR / RUN_NAME

MODE = "train"           # [NEW] "train" = آموزش + val ، "test" = ارزیابی نهایی روی test (فقط یک بار!)
ALLOW_RETEST = False     # [NEW] اگر test قبلاً ارزیابی شده، بدون این True اجازه‌ی تکرار نمی‌دهد
MIN_PIXELS = 50          # [NEW] (فقط diagnose.py) پیکسل پیش‌بینی‌شده روی تصویر normal = false positive

NUM_CLASSES = 2          # 2 = ضایعه/پس‌زمینه ،  3 = پس‌زمینه/خوش‌خیم(سبز)/بدخیم(قرمز)
IMG_W, IMG_H = 384, 256  # هر دو بر 32 بخش‌پذیر
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
SEED = 42
GROUP_COL = None         # اگر اسم ستون بیمار در INFO.csv را می‌دانی اینجا بنویس، وگرنه خودکار پیدا می‌شود
CROP_ARTIFACTS = False   # True = کراپ حاشیه (نشانگرهای گوشه). اول False، بعد مقایسه کن

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "train_log.txt"


def log(msg):                                        # [NEW] چاپ + ذخیره در فایل
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


set_seed(SEED)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# =====================================================================
# 2) DATA: خواندن، تبدیل ماسک رنگی به کلاس، split بر اساس بیمار
# =====================================================================
def color_mask_to_class(rgb: np.ndarray, num_classes: int) -> np.ndarray:
    """rgb: HxWx3 uint8. سبز=خوش‌خیم، قرمز=بدخیم، سیاه=پس‌زمینه."""
    r, g = rgb[..., 0].astype(int), rgb[..., 1].astype(int)
    fg = rgb.max(-1) > 60
    out = np.zeros(rgb.shape[:2], np.uint8)
    if num_classes == 2:
        out[fg] = 1
    else:
        out[fg & (g >= r)] = 1   # سبز
        out[fg & (r > g)] = 2    # قرمز
    return out


def load_pair(name: str):
    img = Image.open(ROOT / "images" / name).convert("RGB")
    msk = Image.open(ROOT / "masks" / name).convert("RGB")
    if CROP_ARTIFACTS:
        w, h = img.size
        box = (0, int(0.04 * h), w, int(0.90 * h))
        img, msk = img.crop(box), msk.crop(box)
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    msk = msk.resize((IMG_W, IMG_H), Image.NEAREST)   # nearest برای ماسک
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)             # uint8 3xHxW
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


def baseline_split(names, groups):
    """[NEW] همان منطق split اولیه، به‌صورت تابع (diagnose.py هم از همین استفاده می‌کند).
    خروجی: سه لیست پایتونی از index ها (train, val, test)."""
    idx = np.arange(len(names)); g = np.array(groups)
    tr, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(idx, groups=g))
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(rest, groups=g[rest]))
    return tr.tolist(), rest[va_rel].tolist(), rest[te_rel].tolist()


class BUSDataset(Dataset):
    def __init__(self, names, cache, train):
        self.names, self.cache, self.train = names, cache, train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        x, y = self.cache[self.names[i]]
        x = x.float() / 255.0
        if self.train:
            if random.random() < 0.5:                       # hflip (هم عکس هم ماسک)
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.8:                       # روشنایی/کنتراست
                x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1)).clamp(0, 1)
            if random.random() < 0.5:                       # شیفت/اسکیل کوچک
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
    print("INFO.csv columns:", df.columns.tolist())
    groups = find_groups(df, names)
    idx = np.arange(len(names))
    if groups is not None:
        tr, va, te = baseline_split(names, groups)
    else:
        print("[split] WARNING: ستون بیمار پیدا نشد → split تصادفی (ممکن است leakage داشته باشد). "
              "GROUP_COL را در CONFIG بگذار.")
        tr, rest = train_test_split(idx, test_size=0.3, random_state=SEED)
        va, te = train_test_split(rest, test_size=0.5, random_state=SEED)
        tr, va, te = tr.tolist(), va.tolist(), te.tolist()

    # [NEW] جدول ترکیب split (فقط نمایش؛ روی آموزش اثری ندارد)
    lab = dict(zip(df["Image"].str.strip(), df["Label"].str.strip()))
    for k, ids in (("train", tr), ("val", va), ("test", te)):
        nm = [names[i] for i in ids]
        cnt = pd.Series([lab.get(n, "?") for n in nm]).value_counts().to_dict()
        log(f"{k:5s} {len(nm):3d} imgs | {len({n.split('_')[0] for n in nm}):2d} patients | {cnt}")

    mk = lambda ids, train: BUSDataset([names[i] for i in ids], cache, train)
    log(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    return (DataLoader(mk(tr, True), BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True),
            DataLoader(mk(va, False), BATCH_SIZE, num_workers=0),
            DataLoader(mk(te, False), BATCH_SIZE, num_workers=0))


# =====================================================================
# 3) MODEL: MobileNetV3-small encoder + U-Net decoder
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
    def __init__(self, num_classes, weights_path=None):
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

    def forward(self, x):
        size = x.shape[-2:]
        c2 = self.s2(x); c4 = self.s4(c2); c8 = self.s8(c4)
        c16 = self.s16(c8); c32 = self.s32(c16)
        d = self.u1(c32, c16); d = self.u2(d, c8); d = self.u3(d, c4); d = self.u4(d, c2)
        return F.interpolate(self.head(d), size=size, mode="bilinear", align_corners=False)


# =====================================================================
# 4) LOSS + METRICS
# =====================================================================
def dice_loss(logits, y, eps=1.0):
    p = logits.softmax(1)
    oh = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps))[1:].mean()      # فقط کلاس‌های ضایعه


def loss_fn(logits, y):
    return F.cross_entropy(logits, y) + dice_loss(logits, y)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    C = NUM_CLASSES
    inter, pred_s, gt_s, tot_loss, n = [torch.zeros(C)] * 0, torch.zeros(C), torch.zeros(C), 0.0, 0
    inter = torch.zeros(C)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        lg = model(x)
        tot_loss += loss_fn(lg, y).item() * len(x); n += len(x)
        p = lg.argmax(1)
        for c in range(1, C):
            inter[c] += ((p == c) & (y == c)).sum().item()
            pred_s[c] += (p == c).sum().item()
            gt_s[c] += (y == c).sum().item()
    dice = (2 * inter[1:] / (pred_s[1:] + gt_s[1:]).clamp(min=1))
    iou = inter[1:] / (pred_s[1:] + gt_s[1:] - inter[1:]).clamp(min=1)
    return tot_loss / n, dice.mean().item(), iou.mean().item()


# =====================================================================
# 5) TRAIN + TEST + VISUALIZATION
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


def current_config():                                # [NEW]
    return {"RUN_NAME": RUN_NAME, "NUM_CLASSES": NUM_CLASSES, "IMG_W": IMG_W, "IMG_H": IMG_H,
            "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "LR": LR, "SEED": SEED,
            "CROP_ARTIFACTS": CROP_ARTIFACTS, "scheduler": "CosineAnnealingLR", "wd": 1e-4}


def save_full(path, model, opt, sched, ep, best, history):   # [NEW] checkpoint کامل برای ادامه دادن
    torch.save({"epoch": ep, "best_score": best, "history": history,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "config": current_config()}, path)


def main():
    log(f"device: {DEVICE} | run: {RUN_NAME}")
    json.dump(current_config(), open(OUT_DIR / "run_config.json", "w"), indent=1)
    tr_dl, va_dl, te_dl = build_loaders()
    model = MBV3SmallUNet(NUM_CLASSES, WEIGHTS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, ckpt = -1.0, OUT_DIR / "best.pth"
    best_full, last_full, history = OUT_DIR / "best_full.pth", OUT_DIR / "last_full.pth", []

    for ep in range(1, EPOCHS + 1):
        model.train(); run, t0 = 0.0, time.time()
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            run += loss.item()
        sched.step()
        vl, vd, vi = evaluate(model, va_dl)
        history.append({"epoch": ep, "train_loss": run / len(tr_dl), "val_loss": vl, "dice": vd, "iou": vi})
        flag = ""
        if vd > best:
            best = vd; flag = "  <- best"
            torch.save(model.state_dict(), ckpt)                          # فقط وزن‌ها (برای diagnose.py)
            save_full(best_full, model, opt, sched, ep, best, history)    # کامل (برای ادامه)
        save_full(last_full, model, opt, sched, ep, best, history)
        json.dump(history, open(OUT_DIR / "history.json", "w"), indent=1)
        log(f"ep {ep:02d}/{EPOCHS} train_loss {run / len(tr_dl):.4f} | val_loss {vl:.4f} dice {vd:.4f} iou {vi:.4f} "
            f"({time.time() - t0:.0f}s){flag}")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    save_preview(model, va_dl, OUT_DIR / "val_preview.png")               # val؛ test دست نمی‌خورد
    json.dump({"best_val_dice": best}, open(OUT_DIR / "final_metrics.json", "w"), indent=1)
    log(f"done. best val dice = {best:.4f} | saved: {ckpt}")
    log("برای ارزیابی نهایی روی test: MODE = 'test' (فقط یک بار، بعد از قفل شدن config).")


def run_test():                                      # [NEW]
    res = OUT_DIR / "test_metrics.json"
    if res.exists() and not ALLOW_RETEST:
        raise SystemExit(f"test قبلاً ارزیابی شده ({res}). تکرار فقط با ALLOW_RETEST=True؛ "
                         "ولی بعد از دیدن test، دیگر تمیز نیست.")
    _, _, te_dl = build_loaders()
    model = MBV3SmallUNet(NUM_CLASSES, WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(OUT_DIR / "best.pth", map_location=DEVICE))
    tl, td, ti = evaluate(model, te_dl)
    log(f"TEST  dice {td:.4f}  iou {ti:.4f}")
    save_preview(model, te_dl, OUT_DIR / "test_preview.png")
    json.dump({"loss": tl, "dice": td, "iou": ti}, open(res, "w"), indent=1)


if __name__ == "__main__":
    main() if MODE == "train" else run_test()