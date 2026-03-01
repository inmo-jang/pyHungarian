import os
import sys
import cProfile
import pstats
import io

# 프로젝트 루트 경로 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import numpy as np
from scipy.optimize import linear_sum_assignment


class Hungarian:
    def __init__(self, weights: np.ndarray, debug: bool = False):
        self.weights = weights
        self.r, self.p = weights.shape
        self.debug = debug

    def solve(self):
        """
        Hungarian algorithm using scipy.optimize.linear_sum_assignment
        """
        # inf를 큰 유한값으로 치환 (scipy는 inf 미지원)
        w = np.where(np.isinf(self.weights), 1e9, self.weights)

        row_ind, col_ind = linear_sum_assignment(w)

        assignment = np.full(self.r, -1, dtype=int)
        assignment[row_ind] = col_ind

        # 실제 weights로 cost 계산 (inf 포함 여부 확인)
        total_cost = float(self.weights[row_ind, col_ind].sum())
        if np.isinf(total_cost):
            return None, None

        if self.debug:
            print(f"Assignment: {assignment}")
            print(f"Total cost: {total_cost}")

        return assignment, total_cost


# Test
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_cases.cases import HUNGARIAN_TEST_CASES

    pr = cProfile.Profile()
    pr.enable()

    for case in HUNGARIAN_TEST_CASES:
        print("=" * 60)
        print(f"Test case: {case['name']}")
        print(case["weights"])

        h = Hungarian(case["weights"].copy(), debug=False)
        assignment, total_cost = h.solve()

        print(f"  → computed total cost: {total_cost}")
        print(f"  → expected optimal cost: {case['optimal_cost']}")
        if total_cost is not None and case['optimal_cost'] is not None:
            print(f"  → diff: {total_cost - case['optimal_cost']}")

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\n" + s.getvalue())
