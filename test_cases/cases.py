# test_cases/cases.py
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    def _optimal_cost(w):
        finite_w = np.where(np.isinf(w), 1e9, w)
        r, c = _lsa(finite_w)
        return float(w[r, c].sum())
except ImportError:
    def _optimal_cost(w):
        return None


def _make_case(name, n, seed, with_inf=False, inf_prob=0.15):
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 100, size=(n, n)).astype(float)
    if with_inf:
        mask = rng.random(size=(n, n)) < inf_prob
        # 각 행/열에 최소 1개의 유효 엣지 보장
        for i in range(n):
            if mask[i].all():
                mask[i, rng.integers(n)] = False
        for j in range(n):
            if mask[:, j].all():
                mask[rng.integers(n), j] = False
        w[mask] = np.inf
    return {"name": name, "weights": w, "optimal_cost": _optimal_cost(w)}


HUNGARIAN_TEST_CASES = [
    # ── 기존 케이스 ──────────────────────────────────────────────
    {
        "name": "case_5x5_with_infty",
        "weights": np.array([
            [5,  np.inf, np.inf, np.inf, np.inf],
            [3,  7,      3,     np.inf, np.inf],
            [6,  np.inf, 2,     np.inf, 4     ],
            [np.inf, np.inf, 5, 6,      2     ],
            [np.inf, 4,      np.inf, 1, 6     ]
        ], dtype=float),
        "optimal_cost": 17.0,
    },
    {
        "name": "case_4x4_basic",
        "weights": np.array([
            [4, 2, 5, 7],
            [8, 3, 10, 8],
            [12, 5, 4, 5],
            [6, 3, 7, 14],
        ], dtype=float),
        "optimal_cost": 19.0,
    },
    {
        "name": "case_3x3_classic",
        "weights": np.array([
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2],
        ], dtype=float),
        "optimal_cost": 5.0,
    },

    # ── 중간 크기 (10x10) ────────────────────────────────────────
    *[_make_case(f"case_10x10_dense_{i}", 10, seed=100+i) for i in range(5)],
    *[_make_case(f"case_10x10_sparse_{i}", 10, seed=200+i, with_inf=True) for i in range(3)],

    # ── 중간 크기 (20x20) ────────────────────────────────────────
    *[_make_case(f"case_20x20_dense_{i}", 20, seed=300+i) for i in range(5)],
    *[_make_case(f"case_20x20_sparse_{i}", 20, seed=400+i, with_inf=True) for i in range(3)],

    # ── 큰 케이스 (50x50) ────────────────────────────────────────
    *[_make_case(f"case_50x50_dense_{i}", 50, seed=500+i) for i in range(3)],
    *[_make_case(f"case_50x50_sparse_{i}", 50, seed=600+i, with_inf=True) for i in range(2)],

    # ── 최대 크기 (100x100) ──────────────────────────────────────
    *[_make_case(f"case_100x100_dense_{i}", 100, seed=700+i) for i in range(2)],
]
