import numpy as np

BYTES_PER_ELEMENT = 4

# Таблица операций. Каждый элемент = одно CUDA-ядро при forward-проходе.
# Поля:
#   f_s2: FLOPs на B*S^2
#   f_b: FLOPs на B (не зависят от S, это Linear-слои головы)
#   m_s2: элементы (чтение+запись) на B*S^2
#   m_b: элементы (чтение+запись) на B
#   w: весов (param count), читаются один раз за проход
OPS = [
    # name         f_s2    f_b     m_s2   m_b    w
    ("conv7x7s2",  2352.0,    0.0,  11.0,   0.0,   4704),
    ("relu",          0.0,    0.0,  16.0,   0.0,      0),
    ("maxpool3s2",    0.0,    0.0,  14.0,   0.0,      0),
    ("conv5x5",    6400.0,    0.0,   6.0,   0.0,  51200),
    ("relu",          0.0,    0.0,   8.0,   0.0,      0),
    ("conv3x3s2",  2304.0,    0.0,   6.0,   0.0,  73728),
    ("relu",          0.0,    0.0,   4.0,   0.0,      0),
    ("conv1x1",    1024.0,    0.0,   6.0,   0.0,  32768),
    ("relu",          0.0,    0.0,   8.0,   0.0,      0),
    ("conv3x3s2",  4608.0,    0.0,   5.0,   0.0, 589824),
    ("relu",          0.0,    0.0,   2.0,   0.0,      0),
    ("conv1x1",    1024.0,    0.0,   3.0,   0.0, 131072),
    ("relu",          0.0,    0.0,   4.0,   0.0,      0),
    ("gap",           2.0,    0.0,   2.0, 512.0,      0),
    ("flatten",       0.0,    0.0,   0.0, 1024.0,     0),
    ("linear512_256", 0.0, 262400.0, 0.0, 768.0, 131328),
    ("relu",          0.0,    0.0,   0.0, 512.0,      0),
    ("linear256_100", 0.0,  51300.0, 0.0, 356.0,  25700),
]

N_OPS = len(OPS)
N_PARAMS = sum(op[5] for op in OPS)


def flops(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    f_s2 = sum(op[1] for op in OPS)   # 17714
    f_b = sum(op[2] for op in OPS)    # 313700
    return B * (f_s2 * S * S + f_b)


def bytes_moved(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    m_s2 = sum(op[3] for op in OPS)   # 95
    m_b = sum(op[4] for op in OPS)    # 3172
    act = B * (m_s2 * S * S + m_b) * BYTES_PER_ELEMENT
    weights = N_PARAMS * BYTES_PER_ELEMENT
    return act + weights


def memory(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    peak_elements = N_PARAMS + B * 17.0 * S * S
    return peak_elements * BYTES_PER_ELEMENT


def latency(image_size, batch, theta):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    total = np.zeros(np.broadcast(S, B).shape, dtype=np.float64)
    for op in OPS:
        f = (op[1] * S * S + op[2]) * B
        b = (op[3] * S * S + op[4]) * B * BYTES_PER_ELEMENT + op[5] * BYTES_PER_ELEMENT
        total = total + np.maximum(np.maximum(theta["t_launch"], f / theta["p"]),
                                   b / theta["bw"])
    return total


def energy(image_size, batch, theta_energy):
    e = N_OPS * theta_energy.get("e_kernel", 0.0)
    e = e + theta_energy["e_flop"] * flops(image_size, batch)
    e = e + theta_energy["e_byte"] * bytes_moved(image_size, batch)
    return e


def latency_terms(image_size, batch, theta):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    t_m = 0.0
    t_c = 0.0
    for op in OPS:
        f = (op[1] * S * S + op[2]) * B
        b = (op[3] * S * S + op[4]) * B * BYTES_PER_ELEMENT + op[5] * BYTES_PER_ELEMENT
        t_m = t_m + b / theta["bw"]
        t_c = t_c + f / theta["p"]
    t_l = N_OPS * theta["t_launch"]
    return float(t_l), float(t_m), float(t_c)


if __name__ == "__main__":
    S = np.array([32, 64, 128, 224, 256, 384, 512])
    B = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])[:, None]
    th = {"t_launch": 6e-6, "bw": 1.5e12, "p": 4e13}
    th_e = {"e_kernel": 2e-6, "e_flop": 1e-10, "e_byte": 1e-13}
    print("параметров сети:", N_PARAMS)
    print("flops(S,B=1):\n", flops(S, 1))
    print("memory(S=224,B) МБ:\n", memory(224, B) / 1e6)
    print("latency(S,B) мс:\n", latency(S, B, th) * 1e3)
    print("bytes(S,B=1) МБ:\n", bytes_moved(S, 1) / 1e6)
    print("energy(S,B=1) мДж:\n", energy(S, 1, th_e) * 1e3)
