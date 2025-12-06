# test_cases/hungarian_cases.py
import numpy as np

HUNGARIAN_TEST_CASES = [
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
        # "optimal_assignment": [0, 2, 4, 3, 1],  # 이런 식으로 추가해도 됨 (나중에)
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
        # optimal assignment: [1, 0, 2]
        #   row0→col1 (1)
        #   row1→col0 (2)
        #   row2→col2 (2)
        #   total = 1 + 2 + 2 = 5
    },    
]
