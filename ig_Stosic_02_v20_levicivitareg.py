from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
regularized Levi-Civita → next

Γ na regularizovanoj Fisher metrike
(za razliku od v2 sirovog Γ=−1/(2p) i v5 Euler-a na istom).

  p_λ = (1−λ)p + λ/n
  g_ii = 1/(p_i + λ)          (Tikhonov / Laplace na dijagonali)
  Γ^i_ii = −1 / (2 (p_i + λ))

Kovarijantna akceleracija:
  a_i = (v1 − v0)_i + Γ_reg,i · v1_i²

Skor: a_reg + blagi (p_λ − p_glob); ban last; next.
CSV ceo, seed=39.
"""



import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
WINDOW = 100
LAMBDA = 0.02  # regularizacija (isti red kao 1/n ≈ 0.026)
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def window_p(draws: np.ndarray, end: int, w: int = WINDOW) -> np.ndarray:
    start = max(0, end - w)
    chunk = draws[start:end]
    cnt = Counter(chunk.reshape(-1).tolist())
    n_slots = max(len(chunk) * FRONT_SELECT, 1)
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def shrink_p(p: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """p_λ = (1−λ)p + λ/n  na simplexu."""
    n = len(p)
    return (1.0 - lam) * p + lam / float(n)


def gamma_raw(p: np.ndarray) -> np.ndarray:
    """Siromani Levi-Civita: Γ = −1/(2p) — samo za dijagnostiku."""
    return -0.5 / np.clip(p, 1e-18, None)


def gamma_reg(p: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """
    Regularizovani LC za g_ii = 1/(p_i + λ):
      Γ^i_ii = −1 / (2 (p_i + λ))
    """
    return -0.5 / (p + lam)


def trajectory_accel_reg(
    draws: np.ndarray, w: int = WINDOW, lam: float = LAMBDA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(draws)
    p0 = window_p(draws, n - 2, w)
    p1 = window_p(draws, n - 1, w)
    p2 = window_p(draws, n, w)
    p0s, p1s, p2s = shrink_p(p0, lam), shrink_p(p1, lam), shrink_p(p2, lam)
    v0 = p1s - p0s
    v1 = p2s - p1s
    g_reg = gamma_reg(p2s, lam)
    g_raw = gamma_raw(p2s)
    a = (v1 - v0) + g_reg * (v1 ** 2)
    return p2s, v1, a, g_reg, g_raw


def number_scores(
    a: np.ndarray,
    p_lam: np.ndarray,
    p_glob: np.ndarray,
    g_reg: np.ndarray,
    g_raw: np.ndarray,
    ban: set[int],
) -> dict[int, float]:
    """
    a_reg + excess na p_λ; + težina gde |Γ_raw − Γ_reg| veliko
    (mesta gde regularizacija menja vezu).
    """
    delta_g = np.abs(g_raw - g_reg)
    max_dg = float(delta_g.max()) if delta_g.max() > 0 else 1.0
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
            continue
        out[n] = float(
            a[i]
            + 0.20 * (p_lam[i] - p_glob[i])
            + 0.10 * (delta_g[i] / max_dg)
        )
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_02_v20(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    p_lam, v1, a, g_reg, g_raw = trajectory_accel_reg(draws)
    p_glob = global_p(draws)
    score = number_scores(a, p_lam, p_glob, g_reg, g_raw, ban)
    combo = predict_next(draws, score, ban)

    print(f"CSV: {csv_path.name}")
    print(
        f"Kola: {len(draws)} | seed={SEED} | WINDOW={WINDOW} | λ={LAMBDA} | ig_02_v20 LC-reg"
    )
    print(f"last: {last.tolist()}")
    print()
    print("=== reg Levi-Civita ===")
    print(
        {
            "v_l2": round(float(np.linalg.norm(v1)), 6),
            "a_l2": round(float(np.linalg.norm(a)), 6),
            "mean_|Γ_reg|": round(float(np.mean(np.abs(g_reg))), 4),
            "mean_|Γ_raw|": round(float(np.mean(np.abs(g_raw))), 4),
            "mean_|ΔΓ|": round(float(np.mean(np.abs(g_raw - g_reg))), 4),
        }
    )
    print()
    ranked = sorted(
        ((n, float(score[n])) for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (a_reg + excess) ===")
    print([(n, round(sc, 6)) for n, sc in ranked[:12]])
    print()
    print("=== next (ig_02_v20 LC-reg) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_02_v20()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | WINDOW=100 | λ=0.02 | ig_02_v20 LC-reg
last: [4, 5, 6, 11, 12, 18, 28]

=== reg Levi-Civita ===
{'v_l2': 0.004427, 'a_l2': 0.006864, 'mean_|Γ_reg|': 11.1254, 'mean_|Γ_raw|': 20.559, 'mean_|ΔΓ|': 9.4336}

=== top12 skor (a_reg + excess) ===
[(26, 0.097271), (21, 0.072653), (36, 0.065842), (2, 0.058611), (15, 0.052908), (25, 0.052662), (10, 0.052631), (32, 0.052557), (17, 0.04821), (22, 0.047681), (37, 0.04762), (23, 0.047343)]

=== next (ig_02_v20 LC-reg) ===
next: [2, x, 15, y, 25, z, 36]
"""
