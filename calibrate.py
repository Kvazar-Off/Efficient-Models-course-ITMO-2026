import csv
import json
import math

import numpy as np
from scipy.optimize import minimize

import equations

CSV_IN = "results/measurements.csv"
JSON_OUT = "results/theta.json"


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            rows.append({
                "S": float(r["S"]),
                "B": float(r["B"]),
                "val": int(r["is_validation"]),
                "lat": float(r["latency_ms"]) / 1000.0,      # с
                "mem": float(r["peak_memory_MB"]) * 1e6,     # байт
                "energy": float(r["energy_J"]),              # Дж
                "spread": float(r["energy_spread_J"]),       # Дж
            })
    return rows


def latency_model(params, S, B):
    theta = {"t_launch": math.exp(params[0]),
             "bw": math.exp(params[1]),
             "p": math.exp(params[2])}
    return equations.latency(S, B, theta)


def fit_latency(rows):
    S = np.array([r["S"] for r in rows])
    B = np.array([r["B"] for r in rows])
    y = np.log(np.array([r["lat"] for r in rows]))

    def loss(p):
        pred = latency_model(p, S, B)
        return np.mean((np.log(pred) - y) ** 2)

    t_min = min(r["lat"] for r in rows)
    p0 = [math.log(t_min / equations.N_OPS), math.log(1.0e12), math.log(2.0e13)]
    res = minimize(loss, p0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-12})
    theta = {"t_launch": math.exp(res.x[0]),
             "bw": math.exp(res.x[1]),
             "p": math.exp(res.x[2])}
    return theta


def fit_energy(rows):
    good = [r for r in rows if r["energy"] > 0 and
            (r["spread"] is None or r["spread"] < 0.5 * r["energy"])]
    S = np.array([r["S"] for r in good])
    B = np.array([r["B"] for r in good])
    y = np.log(np.array([r["energy"] for r in good]))
    F = equations.flops(S, B)
    Q = equations.bytes_moved(S, B)

    def loss(p):
        e = np.exp(p[0]) * F + np.exp(p[1]) * Q
        return np.mean((np.log(e) - y) ** 2)

    res = minimize(loss, [math.log(1e-11), math.log(1e-13)], method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-12})
    return {"e_kernel": 0.0,
            "e_flop": float(np.exp(res.x[0])),
            "e_byte": float(np.exp(res.x[1]))}, len(good), len(rows)


def mape(pred, meas):
    return float(np.mean(np.abs(np.array(pred) / np.array(meas) - 1.0)) * 100.0)


def main():
    rows = load_rows(CSV_IN)
    cal = [r for r in rows if r["val"] == 0]
    val = [r for r in rows if r["val"] == 1]
    print(f"точек: всего {len(rows)}, калибровка {len(cal)}, валидация {len(val)}")

    theta = fit_latency(cal)
    theta_e, n_e_good, n_e_all = fit_energy(cal)
    print("theta latency:", theta)
    print("theta energy :", theta_e, f"(по {n_e_good} точкам из {n_e_all})")

    out = {"latency": theta, "energy": theta_e}
    for name, subset in [("cal", cal), ("val", val)]:
        S = np.array([r["S"] for r in subset])
        B = np.array([r["B"] for r in subset])
        pred = latency_model([math.log(theta["t_launch"]), math.log(theta["bw"]),
                              math.log(theta["p"])], S, B)
        out[f"latency_mape_{name}_pct"] = mape(pred, [r["lat"] for r in subset])

        e_subset = [r for r in subset if r["energy"] > 0 and
                    (not np.isfinite(r["spread"]) or r["spread"] < 0.5 * r["energy"])]
        if e_subset:
            Se = np.array([r["S"] for r in e_subset])
            Be = np.array([r["B"] for r in e_subset])
            pe = equations.energy(Se, Be, theta_e)
            out[f"energy_mape_{name}_pct"] = mape(pe, [r["energy"] for r in e_subset])
            out[f"energy_n_points_{name}"] = len(e_subset)
    out["n_points"] = {"cal": len(cal), "val": len(val)}

    try:
        out["hardware"] = json.load(open("results/metadata.json"))
    except Exception:
        pass

    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("сохранено:", JSON_OUT)


if __name__ == "__main__":
    main()
