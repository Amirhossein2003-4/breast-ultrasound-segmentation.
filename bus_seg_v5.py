
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
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

# =====================================================================
# 1) CONFIG
# =====================================================================
ROOT = Path(r"C:\Users\amir\DL\datasets\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM")
WEIGHTS = Path(r"C:\Users\amir\DL\models\mobilenet_v3_small\mobilenet_v3_small.pth")

RUNS_DIR = Path(r"C:\Users\amir\DL\runs")
RUN_NAME = "bus_uclm_v2_aug"                      # برای هر آزمایش جدید اسم جدید بگذار
OUT_DIR = RUNS_DIR / RUN_NAME

MODE = "train"           # "train" = آموزش + val ، "test" = ارزیابی نهایی روی test (فقط یک بار!)
ALLOW_RETEST = False     # اگر test قبلاً ارزیابی شده، بدون این True اجازه‌ی تکرار نمی‌دهد

# --- split ---
SPLIT_MODE = "baseline"  # "baseline" = همان GroupShuffleSplit اولیه ، "stratified" = split جستجو‌شده
EXPECT_SIZES = (431, 144, 108)       # اندازه‌ی split در baseline؛ None = بدون چک (فقط برای mode=baseline)
SPLIT_FILE = RUNS_DIR / f"bus_uclm_split_{SPLIT_MODE}.json"
SPLIT_SEED = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)     # فقط برای mode=stratified
MIN_PAT_PER_CLASS = 2                # برای mode=stratified شرط است؛ برای baseline فقط هشدار
MIN_PIXELS = 50                      # پیکسل پیش‌بینی‌شده روی تصویر normal = false positive

NUM_CLASSES = 2          # 2 = ضایعه/پس‌زمینه ، 3 = پس‌زمینه/خوش‌خیم/بدخیم
IMG_W, IMG_H = 384, 256  # هر دو بر 32 بخش‌پذیر
BATCH_SIZE = 8
EPOCHS = 40
SEED = 42                # seed آموزش
CROP_ARTIFACTS = False

# --- encoder / decoder (baseline) ---
FREEZE_STAGES = ["s2", "s4", "s8"]   # s16, s32, decoder, head = trainable
DEC_DROPOUT = 0.0
FROZEN_BN_EVAL = True                # BN مراحل فریز در حالت eval بماند (اگر baseline این را نداشت False کن)

# --- optimizer: one-cycle (baseline) ---
ENC_LR, DEC_LR, WEIGHT_DECAY = 3e-4, 1e-3, 1e-4
PCT_START = 0.2
PATIENCE = None
GRAD_REPORT_EVERY = 10

# --- metric برای انتخاب best ---
SELECT_METRIC = "dice_pool"          # "dice_pool" = همان dice قدیمی (pooled)؛ "dice" = per-image روی تصاویر دارای ضایعه

# --- ذخیره ---
SAVE_CKPT = False                    # False = checkpoint روی دیسک ذخیره نمی‌شود (لاگ‌ها همیشه ذخیره می‌شوند)

# --- augmentation متوسط (تنها چیزی که نسبت به baseline عوض شده) ---
AUG_ON = True
AUG_P_AFFINE = 0.7
AUG_ROT = 10.0                       # درجه
AUG_ZOOM = (0.9, 1.2)                # >1 = زوم این
AUG_SHIFT = 0.10                     # نرمال‌شده (بازه‌ی کل = 2)؛ 0.10 یعنی ±5% اندازه‌ی تصویر
AUG_P_INTENSITY = 0.8
AUG_P_GAMMA, AUG_P_NOISE = 0.3, 0.3

RESUME_FROM = None                   # مثال: OUT_DIR / "best_full.pth" (نیاز به SAVE_CKPT=True در اجرای قبلی)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "train_log.txt"
CLASSES = ["Normal", "Benign", "Malignant"]
ENC_STAGES = ["s2", "s4", "s8", "s16", "s32"]
DEC_PARTS = ["u1", "u2", "u3", "u4", "head"]


def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


set_seed(SEED)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# =====================================================================
# 2) DATA
# =====================================================================
def color_mask_to_class(rgb: np.ndarray, num_classes: int) -> np.ndarray:
    """rgb: HxWx3 uint8. سبز=خوش‌خیم، قرمز=بدخیم، سیاه=پس‌زمینه."""
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


def augment(x, y):
    """x: float 3xHxW در [0,1]، y: long HxW. هندسی روی هر دو، شدتی فقط روی عکس."""
    if random.random() < 0.5:                                     # hflip (بدون vertical flip)
        x, y = x.flip(-1), y.flip(-1)
    if random.random() < AUG_P_AFFINE:                            # چرخش + زوم + شیفت
        ang = np.deg2rad(random.uniform(-AUG_ROT, AUG_ROT))
        s = 1.0 / random.uniform(*AUG_ZOOM)
        tx, ty = random.uniform(-AUG_SHIFT, AUG_SHIFT), random.uniform(-AUG_SHIFT, AUG_SHIFT)
        c, sn = float(np.cos(ang)), float(np.sin(ang))
        theta = torch.tensor([[c * s, -sn * s * IMG_H / IMG_W, tx],
                              [sn * s * IMG_W / IMG_H, c * s, ty]], dtype=torch.float32)[None]
        grid = F.affine_grid(theta, (1, 3, IMG_H, IMG_W), align_corners=False)
        x = F.grid_sample(x[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        y = F.grid_sample(y[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0].long()
    if random.random() < AUG_P_INTENSITY:                         # روشنایی/کنتراست
        x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1)).clamp(0, 1)
    if random.random() < AUG_P_GAMMA:
        x = x.clamp(1e-6, 1) ** random.uniform(0.8, 1.25)
    if random.random() < AUG_P_NOISE:
        x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    return x, y


class BUSDataset(Dataset):
    def __init__(self, names, cache, train):
        self.names, self.cache, self.train = names, cache, train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        x, y = self.cache[self.names[i]]
        x = x.float() / 255.0
        if self.train and AUG_ON:
            x, y = augment(x, y)
        return (x - MEAN) / STD, y


# ---------- split ----------
def baseline_split(names, groups):
    """دقیقاً منطق کد اولیه: GroupShuffleSplit(0.3) سپس GroupShuffleSplit(0.5) روی بقیه."""
    idx = np.arange(len(names)); g = np.array(groups)
    tr, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=SPLIT_SEED).split(idx, groups=g))
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.5, random_state=SPLIT_SEED).split(rest, groups=g[rest]))
    return [sorted(tr.tolist()), sorted(rest[va_rel].tolist()), sorted(rest[te_rel].tolist())]


def split_score(parts, labels, groups, tot_frac, n):
    score = 0.0
    for k, idx in enumerate(parts):
        if len(idx) == 0:
            return 1e9
        frac = np.array([sum(labels[i] == c for i in idx) for c in CLASSES]) / len(idx)
        score += np.abs(frac - tot_frac).sum()
        score += 2 * abs(len(idx) / n - SPLIT_FRACS[k])
        if k > 0:
            for c in ("Benign", "Malignant"):
                if len({groups[i] for i in idx if labels[i] == c}) < MIN_PAT_PER_CLASS:
                    score += 10
    return score


def make_split(names, labels, groups):
    """split stratified (بر اساس بیمار) با جستجوی seed."""
    n = len(names)
    tot_frac = np.array([sum(l == c for l in labels) for c in CLASSES]) / n
    by_pat = {}
    for i, g in enumerate(groups):
        by_pat.setdefault(g, []).append(i)
    pats = sorted(by_pat)
    best, best_s = None, 1e18
    for s in range(SPLIT_SEED, SPLIT_SEED + 2000):
        rng = random.Random(s)
        order = pats[:]; rng.shuffle(order)
        parts, k, cnt = [[], [], []], 0, 0
        for p in order:
            if k < 2 and cnt >= sum(SPLIT_FRACS[:k + 1]) * n:
                k += 1
            parts[k] += by_pat[p]; cnt += len(by_pat[p])
        sc = split_score(parts, labels, groups, tot_frac, n)
        if sc < best_s:
            best, best_s = [sorted(p) for p in parts], sc
    return best


def build_loaders():
    names = sorted(p.name for p in (ROOT / "images").glob("*.png"))
    assert names == sorted(p.name for p in (ROOT / "masks").glob("*.png")), "images/masks mismatch"
    df = pd.read_csv(ROOT / "INFO.csv", sep=";")
    lab = dict(zip(df["Image"].str.strip(), df["Label"].str.strip()))
    missing = [n for n in names if n not in lab]
    assert not missing, f"در INFO.csv نیستند: {missing[:5]}"
    labels = [lab[n] for n in names]
    groups = [n.split("_")[0] for n in names]
    log(f"{len(names)} images | {len(set(groups))} patients (باید 38 باشد) | loading into RAM ...")
    cache = {n: load_pair(n) for n in names}

    loaded = SPLIT_FILE.exists()
    if loaded:
        sp = json.load(open(SPLIT_FILE))
        assert set(sum(sp.values(), [])) == set(names), "split فعلی با دیتاست نمی‌خواند؛ فایل split را پاک کن"
    else:
        idx = baseline_split(names, groups) if SPLIT_MODE == "baseline" else make_split(names, labels, groups)
        sp = {k: [names[i] for i in v] for k, v in zip(("train", "val", "test"), idx)}

    sizes = tuple(len(sp[k]) for k in ("train", "val", "test"))
    if SPLIT_MODE == "baseline" and EXPECT_SIZES is not None:
        assert sizes == tuple(EXPECT_SIZES), (f"split با baseline نمی‌خواند: {sizes} به جای {EXPECT_SIZES}. "
                                              "seed یا منطق split اولیه فرق داشته؛ به من بگو.")
    if not loaded:
        json.dump(sp, open(SPLIT_FILE, "w"), indent=1)
    log(f"split mode={SPLIT_MODE} ({'لود از ' + SPLIT_FILE.name if loaded else 'ساخته و ذخیره شد'}) sizes={sizes}")

    pat = {k: {n.split("_")[0] for n in v} for k, v in sp.items()}
    assert not (pat["train"] & pat["val"] or pat["train"] & pat["test"] or pat["val"] & pat["test"]), \
        "LEAKAGE: بیمار مشترک بین splitها!"
    for k, v in sp.items():
        cnt = pd.Series([lab[n] for n in v]).value_counts().to_dict()
        pb = {n.split("_")[0] for n in v if lab[n] == "Benign"}
        pm = {n.split("_")[0] for n in v if lab[n] == "Malignant"}
        log(f"{k:5s} {len(v):3d} imgs | {len(pat[k]):2d} patients (benign {len(pb)}, malignant {len(pm)}) | {cnt}")
        if k != "train" and (len(pb) < MIN_PAT_PER_CLASS or len(pm) < MIN_PAT_PER_CLASS):
            log(f"  ⚠ {k}: بیمار benign یا malignant کم است؛ نتیجه‌ی این split پرنوسان است")

    mk = lambda nm, train: BUSDataset(nm, cache, train)
    return (DataLoader(mk(sp["train"], True), BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True),
            DataLoader(mk(sp["val"], False), BATCH_SIZE, num_workers=0),
            DataLoader(mk(sp["test"], False), BATCH_SIZE, num_workers=0))


# =====================================================================
# 3) MODEL
# =====================================================================
def conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.block = nn.Sequential(conv_bn(in_c + skip_c, out_c), conv_bn(out_c, out_c),
                                   nn.Dropout2d(DEC_DROPOUT) if DEC_DROPOUT > 0 else nn.Identity())

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


def set_train_mode(model):
    model.train()
    if FROZEN_BN_EVAL:
        for n in FREEZE_STAGES:
            getattr(model, n).eval()


def grad_report(model):
    out = []
    for n in ENC_STAGES + DEC_PARTS:
        g = [p.grad.abs().mean().item() for p in getattr(model, n).parameters() if p.grad is not None]
        out.append(f"{n}={np.mean(g):.1e}" if g else f"{n}=frozen")
    log("  grad | " + " ".join(out))


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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
    """dice_pool/iou_pool: pooled روی همه‌ی پیکسل‌ها (همان متریک قدیمی؛ foreground ادغام‌شده).
    dice/iou: میانگین per-image روی تصاویر دارای ضایعه. fp_normal: درصد تصاویر normal با false positive."""
    model.eval()
    tot, n, dices, ious, fps = 0.0, 0, [], [], []
    I = PS = GS = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        lg = model(x)
        tot += loss_fn(lg, y).item() * len(x); n += len(x)
        P, G = (lg.argmax(1) > 0), (y > 0)
        I += (P & G).sum().item(); PS += P.sum().item(); GS += G.sum().item()
        for p, g in zip(P, G):
            inter, ps, gs = (p & g).sum().item(), p.sum().item(), g.sum().item()
            if gs > 0:
                dices.append(2 * inter / (ps + gs)); ious.append(inter / (ps + gs - inter))
            else:
                fps.append(float(ps > MIN_PIXELS))
    nan = float("nan")
    return {"loss": tot / n,
            "dice_pool": 2 * I / max(PS + GS, 1), "iou_pool": I / max(PS + GS - I, 1),
            "dice": float(np.mean(dices)) if dices else nan,
            "iou": float(np.mean(ious)) if ious else nan,
            "fp_normal": float(np.mean(fps)) if fps else nan}


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


CFG_KEYS = ["RUN_NAME", "SPLIT_MODE", "SPLIT_SEED", "NUM_CLASSES", "IMG_W", "IMG_H", "BATCH_SIZE", "EPOCHS", "SEED",
            "CROP_ARTIFACTS", "FREEZE_STAGES", "DEC_DROPOUT", "FROZEN_BN_EVAL", "ENC_LR", "DEC_LR", "WEIGHT_DECAY",
            "PCT_START", "SELECT_METRIC", "AUG_ON", "AUG_P_AFFINE", "AUG_ROT", "AUG_ZOOM", "AUG_SHIFT",
            "AUG_P_INTENSITY", "AUG_P_GAMMA", "AUG_P_NOISE"]


def current_config():
    return {k: globals()[k] for k in CFG_KEYS}


def save_full(path, model, opt, sched, ep, best, history):
    torch.save({"epoch": ep, "best_score": best, "history": history,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "config": current_config()}, path)


def main():
    log(f"device: {DEVICE} | run: {RUN_NAME} | SAVE_CKPT={SAVE_CKPT}")
    json.dump(current_config(), open(OUT_DIR / "run_config.json", "w"), indent=1)
    tr_dl, va_dl, te_dl = build_loaders()
    model = MBV3SmallUNet(NUM_CLASSES, WEIGHTS).to(DEVICE)
    for n in FREEZE_STAGES:
        for p in getattr(model, n).parameters():
            p.requires_grad = False

    enc = [p for n in ENC_STAGES for p in getattr(model, n).parameters() if p.requires_grad]
    dec = [p for n in DEC_PARTS for p in getattr(model, n).parameters()]
    pg, mx = [], []
    if enc:
        pg.append({"params": enc}); mx.append(ENC_LR)
    pg.append({"params": dec}); mx.append(DEC_LR)
    opt = torch.optim.AdamW(pg, lr=1e-3, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=mx, total_steps=EPOCHS * len(tr_dl), pct_start=PCT_START)
    log(f"frozen={FREEZE_STAGES} | trainable params: {sum(p.numel() for p in enc + dec):,} | pct_start={PCT_START}")

    best, bad, start, history = -1.0, 0, 1, []
    ckpt, best_full, last_full = OUT_DIR / "best.pth", OUT_DIR / "best_full.pth", OUT_DIR / "last_full.pth"

    if RESUME_FROM:
        ck = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        for k in ("FREEZE_STAGES", "SELECT_METRIC", "SPLIT_MODE"):
            assert ck["config"][k] == globals()[k], f"{k} با checkpoint فرق دارد"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start, best, history = ck["epoch"] + 1, ck["best_score"], ck["history"]
        log(f"resumed from epoch {ck['epoch']} (best {SELECT_METRIC} {best:.4f})")
    best_state = snapshot(model)

    for ep in range(start, EPOCHS + 1):
        set_train_mode(model); run, t0 = 0.0, time.time()
        for i, (x, y) in enumerate(tr_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            if ep % GRAD_REPORT_EVERY == 0 and i == len(tr_dl) - 1:
                grad_report(model)
            opt.step(); sched.step()
            run += loss.item()
        v = evaluate(model, va_dl)
        history.append({"epoch": ep, "train_loss": run / len(tr_dl), **v})
        score, flag = v[SELECT_METRIC], ""
        if score > best:
            best, bad, flag = score, 0, "  <- best"
            best_state = snapshot(model)
            if SAVE_CKPT:
                torch.save(model.state_dict(), ckpt)
                save_full(best_full, model, opt, sched, ep, best, history)
        else:
            bad += 1
        if SAVE_CKPT:
            save_full(last_full, model, opt, sched, ep, best, history)
        json.dump(history, open(OUT_DIR / "history.json", "w"), indent=1)      # لاگ همیشه ذخیره می‌شود
        log(f"ep {ep:02d}/{EPOCHS} train_loss {run / len(tr_dl):.4f} | val_loss {v['loss']:.4f} "
            f"dice_pool {v['dice_pool']:.4f} iou_pool {v['iou_pool']:.4f} | dice {v['dice']:.4f} "
            f"fp_normal {v['fp_normal']:.2f} ({time.time() - t0:.0f}s){flag}")
        if PATIENCE and bad >= PATIENCE:
            log("early stop"); break

    model.load_state_dict(best_state)
    save_preview(model, va_dl, OUT_DIR / "val_preview.png")                     # فقط val؛ test دست نمی‌خورد
    json.dump({f"best_val_{SELECT_METRIC}": best}, open(OUT_DIR / "final_metrics.json", "w"), indent=1)
    log(f"done. best val {SELECT_METRIC} = {best:.4f}" + ("" if SAVE_CKPT else "  (checkpoint ذخیره نشد: SAVE_CKPT=False)"))


def run_test():
    """ارزیابی نهایی روی test با best_full.pth. فقط یک بار."""
    assert SPLIT_FILE.exists(), "split پیدا نشد؛ test باید با همان split آموزش اجرا شود"
    assert (OUT_DIR / "best_full.pth").exists(), "best_full.pth نیست؛ آن run با SAVE_CKPT=False اجرا شده بود"
    res_path = OUT_DIR / "test_metrics.json"
    if res_path.exists() and not ALLOW_RETEST:
        raise SystemExit(f"test قبلاً ارزیابی شده ({res_path}). تکرار فقط با ALLOW_RETEST=True؛ "
                         "ولی بعد از دیدن test، دیگر تمیز نیست.")
    ck = torch.load(OUT_DIR / "best_full.pth", map_location=DEVICE, weights_only=False)
    for k in ("NUM_CLASSES", "IMG_W", "IMG_H", "CROP_ARTIFACTS", "SPLIT_MODE"):
        assert ck["config"][k] == globals()[k], f"{k} با checkpoint فرق دارد"
    _, _, te_dl = build_loaders()
    model = MBV3SmallUNet(NUM_CLASSES, WEIGHTS).to(DEVICE)
    model.load_state_dict(ck["model"])
    log(f"checkpoint: run={RUN_NAME} epoch={ck['epoch']} best_val_{ck['config']['SELECT_METRIC']}={ck['best_score']:.4f}")
    t = evaluate(model, te_dl)
    log(f"TEST  dice_pool {t['dice_pool']:.4f}  iou_pool {t['iou_pool']:.4f} | dice {t['dice']:.4f}  fp_normal {t['fp_normal']:.2f}")
    save_preview(model, te_dl, OUT_DIR / "test_preview.png")
    json.dump({"epoch": ck["epoch"], "val_score": ck["best_score"], "test": t}, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    main() if MODE == "train" else run_test()