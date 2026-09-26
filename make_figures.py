import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import equations

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)


def load():
    with open("results/theta.json") as f:
        th = json.load(f)
    rows = []
    oom = []
    with open("results/measurements.csv") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                if r["status"] == "OOM":
                    oom.append({"S": float(r["S"]), "B": float(r["B"]),
                                "val": int(r["is_validation"])})
                continue
            rows.append({
                "S": float(r["S"]), "B": float(r["B"]), "val": int(r["is_validation"]),
                "lat": float(r["latency_ms"]),
                "mem": float(r["peak_memory_MB"]),
                "energy": float(r["energy_J"]) * 1e3,     # мДж
                "spread": float(r["energy_spread_J"]) * 1e3,
            })
    return th, rows, oom


def above_gate(r):
    return r["energy"] > 0 and (not np.isfinite(r["spread"]) or r["spread"] < 0.5 * r["energy"])


def scatter_meas(ax, rows, ykey, logy=True):
    cal = [r for r in rows if r["val"] == 0]
    val = [r for r in rows if r["val"] == 1]
    ax.scatter([r["B"] for r in cal], [r[ykey] for r in cal], s=26, c="tab:blue", label="measured (calibration grid)")
    ax.scatter([r["B"] for r in val], [r[ykey] for r in val], s=40, facecolors="none", edgecolors="tab:red", label="measured (unseen validation points)")
    if logy:
        ax.set_yscale("log")
    ax.set_xscale("log", base=2)


def fig_latency_vs_B(th, rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, s in zip(axes, [32, 224, 512]):
        bs = np.logspace(0, np.log10(256), 100)
        t = equations.latency(s, bs, th["latency"]) * 1e3
        ax.plot(bs, t, "-", c="black", label="model $t(S,B,\\theta)$")
        scatter_meas(ax, [r for r in rows if r["S"] == s], "lat")
        ax.set_title(f"latency at S={s}px"); ax.set_xlabel("batch size B")
        ax.set_ylabel("forward time, ms")
    for ax in axes:
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/latency_vs_B.png", dpi=150); plt.close(fig)


def fig_latency_vs_S(th, rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, b in zip(axes, [1, 16, 256]):
        ss = np.logspace(np.log10(32), np.log10(512), 100)
        t = equations.latency(ss, b, th["latency"]) * 1e3
        ax.plot(ss, t, "-", c="black", label="model $t(S,B,\\theta)$")
        scatter_meas2(ax, [r for r in rows if r["B"] == b], "lat")
        ax.set_title(f"latency at B={b}"); ax.set_xlabel("image size S, px")
        ax.set_ylabel("forward time, ms")
    for ax in axes:
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/latency_vs_S.png", dpi=150); plt.close(fig)


def scatter_meas2(ax, pts, ykey):
    cal = [p for p in pts if p["val"] == 0]
    val = [p for p in pts if p["val"] == 1]
    ax.scatter([p["S"] for p in cal], [p[ykey] for p in cal], s=26, c="tab:blue",
               label="measured (calibration grid)")
    ax.scatter([p["S"] for p in val], [p[ykey] for p in val], s=40, facecolors="none",
               edgecolors="tab:red", label="measured (unseen validation points)")
    ax.set_xscale("log"); ax.set_yscale("log")


def fig_latency_map(th, rows):
    ss = np.logspace(np.log10(32), np.log10(512), 80)
    bb = np.logspace(0, np.log10(256), 80)
    S, B = np.meshgrid(ss, bb)
    Z = np.log10(equations.latency(S, B, th["latency"]) * 1e3)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    im0 = axes[0].pcolormesh(bb, ss, Z, shading="auto", cmap="viridis")
    axes[0].scatter([r["B"] for r in rows], [r["S"] for r in rows],
                    c="white", s=12, label="measured points")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size B"); axes[0].set_ylabel("image size S, px")
    axes[0].set_title("predicted latency (model surface over (S, B))")
    fig.colorbar(im0, ax=axes[0], label="log10 time, ms")
    ratio = np.array([r["lat"] / float(equations.latency(r["S"], r["B"], th["latency"]) * 1e3)
                      for r in rows])
    im1 = axes[1].scatter([r["B"] for r in rows], [r["S"] for r in rows],
                          c=np.log10(ratio), s=34, cmap="coolwarm", vmin=-0.5, vmax=0.5,
                          edgecolors="black", linewidths=0.3)
    axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
    axes[1].set_xlabel("batch size B"); axes[1].set_ylabel("image size S, px")
    axes[1].set_title("residual log10(measured/predicted), white = exact")
    fig.colorbar(im1, ax=axes[1], label="log10 ratio")
    fig.tight_layout(); fig.savefig(f"{FIG}/latency_map.png", dpi=150); plt.close(fig)


def fig_pred_vs_meas(th, rows, key, pred_fn, name, xlabel, ylabel, skip_nonpos=True):
    meas, pred, isval = [], [], []
    for r in rows:
        if skip_nonpos and not (r[key] > 0):
            continue
        meas.append(r[key])
        pred.append(pred_fn(r["S"], r["B"]))
        isval.append(r["val"])
    meas = np.array(meas); pred = np.array(pred); isval = np.array(isval)
    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.scatter(pred[isval == 0], meas[isval == 0], s=30, c="tab:blue", label="calibration")
    ax.scatter(pred[isval == 1], meas[isval == 1], s=44, facecolors="none",
               edgecolors="tab:red", label="unseen validation")
    lo = min(meas.min(), pred.min()) * 0.7
    hi = max(meas.max(), pred.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal 1:1")
    for m in (0.5, 2.0):
        ax.plot([lo, hi], [lo * m, hi * m], color="gray", lw=0.6, ls=":")
    mape_cal = np.mean(np.abs(pred[isval == 0] / meas[isval == 0] - 1)) * 100
    mape_val = np.mean(np.abs(pred[isval == 1] / meas[isval == 1] - 1)) * 100
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"predicted {xlabel}"); ax.set_ylabel(f"measured {ylabel}")
    ax.set_title(f"{name}\nMAPE: cal {mape_cal:.1f}% / val {mape_val:.1f}% (dotted 0.5x, 2x)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.png", dpi=150); plt.close(fig)


def fig_memory(th, rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, s in zip(axes, [32, 224, 512]):
        bs = np.logspace(0, np.log10(256), 60)
        ax.plot(bs, equations.memory(s, bs) / 1e6, "k-", label="model (formula)")
        ax.axhline(16e3, color="purple", ls="--", lw=1, label="16 GB GPU (T4-class)")
        scatter_meas(ax, [r for r in rows if r["S"] == s], "mem")
        ax.set_title(f"peak memory at S={s}px"); ax.set_xlabel("batch size B")
        ax.set_ylabel("peak memory, MB"); ax.set_ylim(bottom=1)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/memory_vs_BS.png", dpi=150); plt.close(fig)


def fig_energy_lines(th, rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, s in zip(axes, [32, 224, 512]):
        bs = np.logspace(0, np.log10(256), 60)
        ax.plot(bs, equations.energy(s, bs, th["energy"]) * 1e3, "k-",
                label="model $E(S,B,\\theta_e)$")
        good = [r for r in rows if r["S"] == s and above_gate(r)]
        dead = [r for r in rows if r["S"] == s and not above_gate(r)]
        scatter_meas(ax, good, "energy")
        ax.scatter([r["B"] for r in dead], [max(r["energy"], 1e-3) for r in dead],
                   marker="x", s=18, c="0.7", label="below noise gate")
        ax.set_title(f"energy at S={s}px"); ax.set_xlabel("batch size B")
        ax.set_ylabel("energy per forward, mJ"); ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/energy_vs_BS.png", dpi=150); plt.close(fig)


# ---------------- 7: режимы ----------------

def fig_regimes(th, rows):
    theta = th["latency"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    pts = []
    for r in rows:
        tl, tm, tc = equations.latency_terms(r["S"], r["B"], theta)
        dom = "launch" if tl >= max(tm, tc) else ("memory" if tm >= tc else "compute")
        pts.append((r["B"], r["S"], dom))
    colors = {"launch": "tab:blue", "memory": "tab:orange", "compute": "tab:red"}
    for dom, c in colors.items():
        sel = [(b, s) for (b, s, d) in pts if d == dom]
        if sel:
            axes[0].scatter([x[0] for x in sel], [x[1] for x in sel], c=c, s=40,
                            label=f"{dom}-bound")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size B"); axes[0].set_ylabel("image size S, px")
    axes[0].set_title("dominant cost term of the model at each measured (S, B)")
    axes[0].legend(fontsize=7, loc="upper left")
    # (b) вклад трёх компонент в предсказанное время при S=224
    bs = np.logspace(0, np.log10(256), 60)
    arr = np.array([list(equations.latency_terms(224, b, theta)) for b in bs]) * 1e3  # мс
    model_line = np.array([float(equations.latency(224, b, theta)) for b in bs]) * 1e3
    axes[1].plot(bs, model_line, "k-", lw=2, label="model = Σ_i max(...)")
    axes[1].plot(bs, arr[:, 0], "--", c="tab:blue", label="N·t_launch")
    axes[1].plot(bs, arr[:, 1], "--", c="tab:orange", label="Bytes/BW")
    axes[1].plot(bs, arr[:, 2], "--", c="tab:red", label="FLOPs/P")
    good = [r for r in rows if r["S"] == 224]
    scatter_meas(axes[1], good, "lat")
    axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
    axes[1].set_xlabel("batch size B"); axes[1].set_ylabel("time, ms")
    axes[1].set_title("latency components at S=224px (max-term per kernel)")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/regimes.png", dpi=150); plt.close(fig)


def fig_oom(rows, oom):
    """Граница OOM: формула memory(S,B) против фактических статусов прогона
    с бюджетом --mem-limit-gb. Предсказанная граница: B = (L/4 - N_params)/(17*S^2)."""
    if not oom:
        print("нет OOM-точек — график границы не строится")
        return
    try:
        with open("results/metadata.json") as f:
            limit_gb = json.load(f).get("mem_limit_gb", 0)
    except Exception:
        limit_gb = 0
    if not limit_gb:
        print("в metadata нет mem_limit_gb — график границы не строится")
        return
    L4 = limit_gb * 1e9 / 4.0          # предельное число fp32-элементов
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ss = np.linspace(32, 512, 200)
    bb_edge = np.clip((L4 - equations.N_PARAMS) / (17.0 * ss * ss), 0, None)
    ax.plot(ss, bb_edge, "k-", label=f"предсказанная граница OOM (memory(S,B) = {limit_gb} ГБ)")
    bb_edge2 = np.clip((L4 - equations.N_PARAMS - 10.0e6) / (17.0 * ss * ss), 0, None)
    ax.plot(ss, bb_edge2, "k--", lw=0.8, color="gray", label="граница со сдвигом на workspace +35 МБ")
    ax.scatter([r["S"] for r in rows], [r["B"] for r in rows], s=30, c="tab:green",
               label="прошло (ok)")
    ax.scatter([r["S"] for r in oom], [r["B"] for r in oom], s=42, c="tab:red",
               marker="x", label="OOM")
    ax.set_xlabel("image size S, px"); ax.set_ylabel("batch size B")
    ax.set_title(f"OOM-граница: формула против фактических статусов (бюджет {limit_gb} ГБ)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/oom_boundary.png", dpi=150); plt.close(fig)


def main():
    th, rows, oom = load()
    fig_latency_vs_B(th, rows)
    fig_latency_vs_S(th, rows)
    fig_latency_map(th, rows)
    fig_pred_vs_meas(th, rows, "lat",
                     lambda s, b: equations.latency(s, b, th["latency"]) * 1e3,
                     "latency_pred_vs_measured", "ms", "ms")
    fig_memory(th, rows)
    fig_pred_vs_meas(th, rows, "mem",
                     lambda s, b: equations.memory(s, b) / 1e6,
                     "memory_pred_vs_measured", "MB", "MB")
    fig_energy_lines(th, rows)
    fig_pred_vs_meas(th, [r for r in rows if above_gate(r)], "energy",
                     lambda s, b: equations.energy(s, b, th["energy"]) * 1e3,
                     "energy_pred_vs_measured", "mJ", "mJ")
    fig_regimes(th, rows)
    fig_oom(rows, oom)
    print("фигуры сохранены в", FIG)


if __name__ == "__main__":
    main()
