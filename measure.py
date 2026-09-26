import argparse
import csv
import os
import random
import threading
import time

import numpy as np
import torch
import pynvml

import models
import equations

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

ENERGY_SEC = 4.0 
N_ATTEMPTS = 2

def make_grid(seed):
    rng = random.Random(seed)
    all_s = list(range(32, 513, 16))
    extra_s_choices = [s for s in all_s if s not in BASE_S]
    extra_s = sorted(rng.sample(extra_s_choices, 4))
    extra_b = []
    while len(extra_b) < 3:
        b = rng.randint(1, 256)
        if (b & (b - 1)) != 0 and b not in extra_b:
            extra_b.append(b)
    all_S = sorted(set(BASE_S) | set(extra_s))
    all_B = sorted(set(BASE_B) | set(extra_b))
    grid = []
    for s in all_S:
        for b in all_B:
            is_val = 1 if (s in extra_s or b in extra_b) else 0
            grid.append((s, b, is_val))
    return grid, extra_s, extra_b


class PowerSampler(threading.Thread):
    def __init__(self, nvml_handle, period_s=0.003):
        super().__init__(daemon=True)
        self.handle = nvml_handle
        self.period = period_s
        self.samples = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            t = time.perf_counter()
            p = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0     
            e = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)  
            u = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            self.samples.append((t, p, e, u))
            time.sleep(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)


def get_nvml_handle():
    pynvml.nvmlInit()
    torch.cuda.init()
    my_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        if pynvml.nvmlDeviceGetUUID(h) == my_uuid:
            return h
    raise RuntimeError("GPU не найден через NVML")


def measure_one(model, x, sampler):
    with torch.inference_mode():
        for _ in range(10):
            model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(x)
        torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()

    work = x.numel()
    n_iter = int(min(500, max(10, 3e9 / max(work, 1))))
    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(n_iter):
            start.record()
            model(x)
            stop.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(stop))  # мс
    latency_ms = float(np.median(times))

    def window_counter_power(ta, tb):
        seg = [(ts, e) for (ts, p, e, u) in sampler.samples if ta <= ts <= tb]
        if len(seg) < 3 or seg[-1][0] <= seg[0][0]:
            return None
        return ((seg[-1][1] - seg[0][1]) / (seg[-1][0] - seg[0][0]) / 1000.0,
                seg[-1][0] - seg[0][0])

    n_e = max(5, int(ENERGY_SEC / max(latency_ms / 1000.0, 1e-4)))
    per_burst = []
    base_last = float("nan")
    for _ in range(N_ATTEMPTS):
        time.sleep(2.5)
        t_deadline = time.perf_counter() + 3.0
        quiet = False
        while time.perf_counter() < t_deadline:
            time.sleep(0.5)
            recent = [u for (ts, p, e, u) in sampler.samples[-25:]]
            if len(recent) >= 10 and max(recent) == 0:
                quiet = True
                break
        if not quiet:
            continue
        tb0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_e):
                model(x)
            torch.cuda.synchronize()
        tb1 = time.perf_counter()
        base = window_counter_power(tb0 - 1.0, tb0)
        cur = window_counter_power(tb0, tb1)
        if base is None or cur is None:
            continue
        d_power = cur[0] - base[0]
        per_burst.append(d_power * cur[1] / n_e)
        base_last = base[0]

    if per_burst:
        energy_j = max(0.0, float(np.median(per_burst)))
        mean_power = float("nan")
    else:
        energy_j = float("nan")
        mean_power = float("nan")
    base = base_last

    return {
        "latency_ms": latency_ms,
        "peak_memory_MB": peak_bytes / 1e6,
        "energy_J": energy_j,
        "energy_spread_J": float(np.std(per_burst)) if len(per_burst) > 1 else float("nan"),
        "n_bursts_ok": len(per_burst),
        "power_base_W": base,
        "n_iter": n_iter,
        "n_iter_energy": n_e,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/measurements.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="для отладки")
    ap.add_argument("--resume", action="store_true", help="дописывать в существующий CSV, пропуская уже измеренные (S,B)")
    ap.add_argument("--mem-limit-gb", type=float, default=2.0,
                    help="бюджет памяти процесса в ГБ (set_per_process_memory_fraction); 0 = без лимита")
    args = ap.parse_args()

    device = "cuda"
    grid, extra_s, extra_b = make_grid(args.seed)
    if args.quick:
        grid = [(32, 1, 0), (64, 8, 0), (224, 64, 0), (512, 256, 0)]

    done = set()
    import csv
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for r in csv.DictReader(f):
                done.add((int(r["S"]), int(r["B"])))
    grid = [g for g in grid if (g[0], g[1]) not in done]

    # Эмуляция low-grade GPU: ограничиваем процесс аллокатора заданным бюджетом.
    # Дробь считается от ПОЛНОЙ памяти устройства, поэтому 2.0 ГБ означают
    # 2.0 ГБ и на H100 (96 ГБ), и на T4 (16 ГБ). Сверх бюджета torch бросает
    # torch.cuda.OutOfMemoryError — его ловим ниже и пишем строку status=OOM.
    if args.mem_limit_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory   # байты
        frac = min(1.0, args.mem_limit_gb * 1e9 / total)
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"лимит памяти процесса: {args.mem_limit_gb} ГБ "
              f"(fraction={frac:.5f} от {total/1e9:.1f} ГБ)")

    torch.manual_seed(0)
    model = models.build_model().to(device).eval()
    handle = get_nvml_handle()
    sampler = PowerSampler(handle)
    sampler.start()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"дополнительные S (валидация): {extra_s}")
    print(f"дополнительные B (валидация): {extra_b}")
    if done:
        print(f"resume: {len(done)} точек уже измерено, осталось {len(grid)}")
    print(f"конфигураций к прогону: {len(grid)}")

    import json
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "driver_cuda": torch.version.cuda,
            "torch": torch.__version__,
            "seed": args.seed,
            "mem_limit_gb": args.mem_limit_gb,
            "extra_s": extra_s,
            "extra_b": extra_b,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    fieldnames = ["S", "B", "is_validation", "status", "latency_ms",
                  "peak_memory_MB", "energy_J", "energy_spread_J", "n_bursts_ok", "power_base_W",
                  "n_iter", "n_iter_energy", "pred_flops", "pred_bytes_MB",
                  "pred_mem_MB"]
    append = bool(done)          # resume: дописываем в конец без нового заголовка
    with open(args.out, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for (s, b, is_val) in grid:
            row = {"S": s, "B": b, "is_validation": is_val}
            try:
                x = torch.randn(b, 3, s, s, device=device)
                res = measure_one(model, x, sampler)
                del x
                # возвращаем аллокатор в чистое состояние: бюджет (set_per_process_
                # memory_fraction) считается по ЗАРЕЗЕРВИРОВАННЫМ сегментам, и
                # закешированные блоки прошлой конфигурации сдвигают границу OOM
                # вниз без всякой физической причины. Пик каждой точки меряется
                # с нуля.
                torch.cuda.empty_cache()
                row.update(res)
                row["status"] = "ok"
            except torch.cuda.OutOfMemoryError:
                for k in fieldnames:
                    if k not in row:
                        row[k] = ""
                row["status"] = "OOM"
                torch.cuda.empty_cache()
            row["pred_flops"] = float(equations.flops(s, b))
            row["pred_bytes_MB"] = float(equations.bytes_moved(s, b)) / 1e6
            row["pred_mem_MB"] = float(equations.memory(s, b)) / 1e6
            writer.writerow(row)
            f.flush()
            if row["status"] == "ok":
                print(f"S={s:4d} B={b:4d} {'val' if is_val else 'cal'} "
                      f"t={row['latency_ms']:8.3f} ms  mem={row['peak_memory_MB']:9.1f} MB "
                      f"E={1000*row['energy_J']:8.3f} mJ", flush=True)
            else:
                print(f"S={s:4d} B={b:4d} {'val' if is_val else 'cal'} OOM", flush=True)
    sampler.stop()
    print("готово:", args.out)


if __name__ == "__main__":
    main()
