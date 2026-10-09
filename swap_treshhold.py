
import numpy as np
import torch
import bus_seg as B

CKPT = r"C:\Users\amir\DL\runs\baseline_rerun\best.pth"
THRS = [0.5, 0.4, 0.3, 0.2, 0.1]
assert B.NUM_CLASSES == 2

names = sorted(p.name for p in (B.ROOT / "images").glob("*.png"))
groups = [n.split("_")[0] for n in names]
_, va, _unused_test = B.baseline_split(names, groups)

model = B.MBV3SmallUNet(B.NUM_CLASSES, B.WEIGHTS).to(B.DEVICE)
ck = torch.load(CKPT, map_location=B.DEVICE, weights_only=False)
model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
model.eval()

res = {t: {"dice": [], "fp": [], "empty": 0} for t in THRS}
signal = []
for i in va:
    x_u8, y = B.load_pair(names[i])
    x = ((x_u8.float() / 255.0 - B.MEAN) / B.STD)[None].to(B.DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0, 1].cpu()          # احتمال ضایعه
    gt = y > 0
    if gt.any():
        signal.append(p[gt].max().item())                   # بیشترین احتمال داخل ناحیه‌ی واقعی ضایعه
    for t in THRS:
        pr = p > t
        inter, ps, gs = (pr & gt).sum().item(), pr.sum().item(), gt.sum().item()
        if gs > 0:
            res[t]["dice"].append(2 * inter / (ps + gs)); res[t]["empty"] += int(ps == 0)
        else:
            res[t]["fp"].append(float(ps > B.MIN_PIXELS))

print("thr   dice(lesion)   empty_pred(lesion)   fp_normal")
for t in THRS:
    n = len(res[t]["dice"])
    print(f"{t:.1f}   {np.mean(res[t]['dice']):.3f}          {res[t]['empty']}/{n}               {np.mean(res[t]['fp']):.2f}")
print("max P(lesion) داخل GT، quantile 10/25/50/75/90:", np.quantile(signal, [.1, .25, .5, .75, .9]).round(2))