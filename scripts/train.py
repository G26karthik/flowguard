"""Train an FNO on PDEBench Darcy coefficient fields.

--target steady : unit-source steady solutions re-solved with the matched discretisation (data/cache)
--target label  : PDEBench's own beta=1 labels (includes the unconverged samples)

Split is fixed (seed 0): 8000 train / 1000 val / 1000 test, saved to results/split.npz.
Usage: python scripts/train.py --target steady --seed 0 --epochs 100 --out runs/fno_steady_s0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowguard.model import FNO2d  # noqa: E402

RAW = "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"
STEADY = "data/cache/2D_DarcyFlow_beta1.0_Train_steady.npy"

p = argparse.ArgumentParser()
p.add_argument("--target", choices=["steady", "label"], default="steady")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--modes", type=int, default=12)
p.add_argument("--width", type=int, default=32)
p.add_argument("--out", required=True)
p.add_argument("--init", help="checkpoint to fine-tune from (keeps its architecture and output scale)")
p.add_argument("--extra", help="npz with a, u (unit source) appended to training; last 300 held out as val_extra")
p.add_argument("--disc", default="arith,ghost", choices=["arith,ghost", "harm,face"],
               help="discretisation of the steady targets (harm,face: data/cache/*_steady_harm_face.npy)")
args = p.parse_args()
DISC = args.disc.split(",")
if DISC != ["arith", "ghost"]:
    STEADY = STEADY.replace("_steady.npy", f"_steady_{DISC[0]}_{DISC[1]}.npy")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

split_path = Path("results/split.npz")
if not split_path.exists():
    perm = np.random.default_rng(0).permutation(10000)
    np.savez(split_path, train=np.sort(perm[:8000]), val=np.sort(perm[8000:9000]), test=np.sort(perm[9000:]))
split = np.load(split_path)

with h5py.File(RAW, "r") as h:
    nu = h["nu"][:]
    label = h["tensor"][:, 0]
steady = np.load(STEADY)
y_all = steady if args.target == "steady" else label

x = torch.from_numpy(np.log10(nu)).unsqueeze(1).to(dev)  # log-permeability input: -1 / 0 for PDEBench
y = torch.from_numpy(y_all).unsqueeze(1).to(dev)
y_scale = float(y[split["train"]].std())
tr, va = torch.from_numpy(split["train"]).to(dev), torch.from_numpy(split["val"]).to(dev)
va_extra = None
if args.extra:
    ex = np.load(args.extra)
    n0, ne = len(x), len(ex["a"])
    x = torch.cat([x, torch.from_numpy(np.log10(ex["a"])).unsqueeze(1).to(dev)])
    y = torch.cat([y, torch.from_numpy(ex["u"]).unsqueeze(1).to(dev)])
    # extra fields are generated family by family: shuffle before holding out, so val_extra covers all families
    perm_ex = torch.from_numpy(np.random.default_rng(1).permutation(ne) + n0).to(dev)
    tr, va_extra = torch.cat([tr, perm_ex[:-300]]), perm_ex[-300:]

model = FNO2d(modes=args.modes, width=args.width).to(dev)
if args.init:
    ck0 = torch.load(args.init, map_location=dev, weights_only=False)
    model = FNO2d(**ck0["cfg"]).to(dev)
    model.load_state_dict(ck0["state_dict"])
    y_scale = ck0["y_scale"]
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps = args.epochs * (len(tr) // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)


def rel_l2(pred, true):
    return (pred - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)


@torch.no_grad()
def evaluate(idx):
    model.eval()
    errs = torch.cat([rel_l2(model(x[idx[i:i + 64]]) * y_scale, y[idx[i:i + 64]]) for i in range(0, len(idx), 64)])
    model.train()
    return errs


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log = []
t0 = time.time()
for ep in range(args.epochs):
    perm = tr[torch.randperm(len(tr), device=dev)]
    tot = 0.0
    for i in range(0, len(perm) - args.batch + 1, args.batch):
        b = perm[i:i + args.batch]
        loss = rel_l2(model(x[b]), y[b] / y_scale).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    if ep % 5 == 4 or ep == args.epochs - 1:
        v = evaluate(va)
        log.append({"epoch": ep + 1, "train_rel_l2": tot / (len(perm) // args.batch), "val_rel_l2_mean": float(v.mean()), "time_s": time.time() - t0})
        if va_extra is not None:
            log[-1]["val_extra_rel_l2_mean"] = float(evaluate(va_extra).mean())
        print(json.dumps(log[-1]), flush=True)

torch.save({"state_dict": model.state_dict(), "cfg": model.cfg, "y_scale": y_scale, "input": "log10(a)",
            "target": args.target, "seed": args.seed, "discretisation": DISC,
            "init": args.init, "extra": args.extra}, out / "model.pt")

# test metrics against both references, so models trained on either target are comparable
te = torch.from_numpy(split["test"]).to(dev)
model.eval()
with torch.no_grad():
    pred = torch.cat([model(x[te[i:i + 64]]) * y_scale for i in range(0, len(te), 64)])
ys, yl = torch.from_numpy(steady[split["test"]]).unsqueeze(1).to(dev), torch.from_numpy(label[split["test"]]).unsqueeze(1).to(dev)
e_s, e_l = rel_l2(pred, ys).cpu().numpy(), rel_l2(pred, yl).cpu().numpy()
q = lambda e: {"mean": float(e.mean()), "median": float(np.median(e)), "p90": float(np.quantile(e, 0.9)), "max": float(e.max())}
metrics = {"args": vars(args), "test_vs_steady": q(e_s), "test_vs_pdebench_label": q(e_l), "train_time_s": time.time() - t0, "log": log}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
np.save(out / "test_rel_l2_vs_steady.npy", e_s)
print(json.dumps({k: metrics[k] for k in ("test_vs_steady", "test_vs_pdebench_label")}, indent=2))
