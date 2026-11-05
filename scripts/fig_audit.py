"""Figure: PDEBench Darcy labels vs the steady state. Writes results/fig_label_audit.png."""
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

betas = ["0.01", "1.0", "100.0"]
dev = {b: np.load(f"results/labelcheck_2D_DarcyFlow_beta{b}_Train.npz")["dev"] for b in betas}
dec = np.load("results/decay_2D_DarcyFlow_beta1.0_Train.npz")
i = int(np.argsort(dev["1.0"])[-50])  # a clearly unconverged beta=1 sample (50th worst of 10k)
with h5py.File("data/raw/2D_DarcyFlow_beta1.0_Train.hdf5", "r") as h:
    nu, lab = h["nu"][i], h["tensor"][i, 0]
st = np.load("data/cache/2D_DarcyFlow_beta1.0_Train_steady.npy", mmap_mode="r")[i]

fig, ax = plt.subplots(1, 5, figsize=(17, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.3, 1.3]})
ax[0].imshow(nu.T, origin="lower", cmap="cividis"); ax[0].set_title(f"a(x), sample {i}")
vmax = max(lab.max(), st.max())
ax[1].imshow(lab.T, origin="lower", vmin=0, vmax=vmax); ax[1].set_title("PDEBench label (beta=1)")
ax[2].imshow(st.T, origin="lower", vmin=0, vmax=vmax); ax[2].set_title(f"steady solve, {100 * dev['1.0'][i]:.0f}% apart")
for a in ax[:3]:
    a.set_xticks([]); a.set_yticks([])
d = (lab - st).T
ax[3].imshow(d, origin="lower", cmap="RdBu_r", vmin=-abs(d).max(), vmax=abs(d).max()); ax[3].set_title("label - steady"); ax[3].set_xticks([]); ax[3].set_yticks([])
for b, c in zip(betas, ["#B0144F", "#0F7480", "#888888"]):
    ax[4].hist(np.log10(np.maximum(dev[b], 1e-5)), bins=60, histtype="step", color=c, label=f"beta={b}")
ax[4].set_xlabel("log10 relative L2, label vs steady"); ax[4].set_ylabel("samples"); ax[4].legend(frameon=False)
fig.tight_layout()
fig.savefig("results/fig_label_audit.png", dpi=110)

fig, axd = plt.subplots(figsize=(4.5, 3.4))
axd.scatter(dec["lam"], dec["dev"], s=8, color="#1D2B34")
axd.set_yscale("log"); axd.set_xlabel("smallest eigenvalue lambda_1 of the operator"); axd.set_ylabel("label vs steady (rel. L2)")
axd.set_title("Slow-decaying fields keep the initial condition")
fig.tight_layout()
fig.savefig("results/fig_decay.png", dpi=110)
print("sample", i, "dev", dev["1.0"][i])
