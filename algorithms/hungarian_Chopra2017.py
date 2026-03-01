import os
import sys
import cProfile
import pstats
import io

# 프로젝트 루트 경로 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import numpy as np
import random

class Hungarian:
    def __init__(self, weights: np.ndarray, debug: bool = False):
        self.weights = weights
        self.r, self.p = weights.shape
        self.r_label = np.min(weights, axis=1)
        self.p_label = np.zeros(self.p, dtype=float)
        self.M = []
        self.Ey = set()
        self.Rc = set()
        self.Pc = set()
        self.Vc = set()
        self.E_cand = set()
        self.delta = None
        self.debug = debug


    def calculate_slack(self, i, j):
        slack = self.weights[i, j] - self.r_label[i] - self.p_label[j]
        return slack

    def build_equality_edges(self, eps=1e-10):
        """Ey = {(i,j) | weights[i,j] == r_label[i]+p_label[j]}"""
        self.Ey = set()
        for i in range(self.r):
            for j in range(self.p):
                if np.isinf(self.weights[i, j]):
                    continue
                if abs(self.calculate_slack(i, j)) <= eps:
                    self.Ey.add((i, j))
        return self.Ey

    def bmp(self, r, match_r, match_p, adj, visited):
        """Bipartite matching using Kuhn's algorithm (DFS approach)."""
        if visited[r]:
            return False
        visited[r] = True
        for p in adj[r]:
            if match_p[p] == -1 or self.bmp(match_p[p], match_r, match_p, adj, visited):
                match_r[r] = p
                match_p[p] = r
                return True
        return False

    def find_matching_and_cover(self, r, p):
        """
        Find maximum matching and minimum vertex cover using König's theorem.
        Automatically provides maximum diversity by:
        - Random row ordering for matching
        - Random selection between standard and alternative vertex cover methods
        
        Returns:
            tuple: (matching, (Rc, Pc), method) where:
                - matching: list of (r,p) pairs
                - Rc, Pc: row and column indices in vertex cover
                - method: 'standard' or 'alternative' indicating which method was used
        """
        # Update labels
        self.r_label = r
        self.p_label = p
        
        # Build equality edges with updated labels
        self.build_equality_edges()
        
        # Create adjacency list for equality edges
        adj = [[] for _ in range(self.r)]
        for r_idx, p_idx in self.Ey:
            adj[r_idx].append(p_idx)
        
        # Initialize matching arrays
        match_r = [-1] * self.r  # match_r[r] = p means row r is matched to column p
        match_p = [-1] * self.p  # match_p[p] = r means column p is matched to row r
        
        # Random row ordering for diversity
        rows = list(range(self.r))
        random.shuffle(rows)
        
        # Find maximum matching using randomized row order
        for r_idx in rows:
            visited = [False] * self.r
            self.bmp(r_idx, match_r, match_p, adj, visited)
        
        # Extract matching pairs
        matching = []
        for r_idx in range(self.r):
            if match_r[r_idx] != -1:
                matching.append((r_idx, match_r[r_idx]))
        
        # Calculate vertex cover using König's theorem
        # Randomly choose between standard and alternative methods
        vertex_cover_method = random.choice(['standard', 'alternative'])
        
        if vertex_cover_method == 'standard':
            # Standard: Start from unmatched rows, find alternating paths
            unmatched_r = [r for r in range(self.r) if match_r[r] == -1]
            reachable_r = set()
            reachable_p = set()
            
            def dfs_from_unmatched_rows(r):
                if r in reachable_r:
                    return
                reachable_r.add(r)
                for p in adj[r]:
                    if p not in reachable_p:
                        reachable_p.add(p)
                        if match_p[p] != -1:
                            dfs_from_unmatched_rows(match_p[p])
            
            for r in unmatched_r:
                dfs_from_unmatched_rows(r)
            
            # Vertex cover: unreachable rows + reachable columns
            Rc = [r for r in range(self.r) if r not in reachable_r]
            Pc = list(reachable_p)
            
        else:  # alternative
            # Alternative: Start from unmatched columns, find alternating paths
            unmatched_p = [p for p in range(self.p) if match_p[p] == -1]
            reachable_r = set()
            reachable_p = set()
            
            def dfs_from_unmatched_cols(p):
                if p in reachable_p:
                    return
                reachable_p.add(p)
                for r in range(self.r):
                    if p in adj[r] and r not in reachable_r:
                        reachable_r.add(r)
                        if match_r[r] != -1:
                            dfs_from_unmatched_cols(match_r[r])
            
            for p in unmatched_p:
                dfs_from_unmatched_cols(p)
            
            # Vertex cover: reachable rows + unreachable columns
            Rc = list(reachable_r)
            Pc = [p for p in range(self.p) if p not in reachable_p]
            Vc = set(Rc).union(set(Pc))
        
        # Update class attributes with the computed vertex cover
        self.Rc = set(Rc)
        self.Pc = set(Pc)
        self.M = matching
        self.Vc = self.Rc.union(self.Pc)
        
        return matching, (Rc, Pc)

    def step_1_a(self):
        """
        Step 1 (a): Find minimum slack edges from uncovered rows(R\Rc) to uncovered columns(P\Pc)
        """
        E_cand = set()
        
        # uncovered rows: vertex cover에 포함되지 않은 행들(R\Rc)
        uncovered_rows = [i for i in range(self.r) if i not in self.Rc]
        # uncovered columns: vertex cover에 포함되지 않은 열들(P\Pc)
        uncovered_cols = [j for j in range(self.p) if j not in self.Pc]
        
        if not uncovered_rows or not uncovered_cols:
            self.E_cand = E_cand
            return E_cand
        
        # 모든 uncovered row에서 uncovered column으로 가는 최소 slack 찾기
        min_global_slack = np.inf
        best_edges = []
        
        for i in uncovered_rows:
            for j in uncovered_cols:
                if np.isinf(self.weights[i, j]):  # 불가용 edge는 제외
                    continue
                    
                slack = self.calculate_slack(i, j)
                if slack < min_global_slack:
                    min_global_slack = slack
                    best_edges = [(i, j)]
                elif slack == min_global_slack:
                    best_edges.append((i, j))
        
        # 최소 slack을 가진 모든 edge들을 후보로 추가
        E_cand = set(best_edges)
        
        self.E_cand = E_cand
        return E_cand

    def step_1_b(self):
        """
        Step 1b: Calculate delta and update labels
        """
        E_cand = self.E_cand
        
        # E_cand에서 최소 slack을 가진 edge와 vertex 쌍 찾기
        min_slack = np.inf
        
        for i, j in E_cand:
            slack = self.calculate_slack(i, j)
            if slack < min_slack:
                min_slack = slack
        
        delta = min_slack
        self.delta = delta
        self.update_labels(delta)
        
        return delta

    def update_labels(self, delta):
        """
        라벨 업데이트 (Hungarian algorithm step)
        """
        if delta is None:
            raise ValueError("Delta is None, cannot update labels.")
        
        uncovered_p = [j for j in range(self.p) if j not in self.Pc]
        
        # Rc에 속한 행들의 r_label을 delta만큼 감소
        for i in self.Rc:
            self.r_label[i] -= delta
            
        # P\Pc에 속한 열들의 p_label을 delta만큼 증가  
        for j in uncovered_p:
            self.p_label[j] += delta
            
        return self.r_label, self.p_label

    def solve(self, max_iter=5):
        """
        Hungarian algorithm debug version with limited iterations
        """
        n = self.r
        iteration = 0
        
        if self.debug:
            print("=== Hungarian Algorithm Debug (While Loop) ===")
            print(f"Initial labels: r={self.r_label}, p={self.p_label}")
            print()
        
        # 초기 Ey, Matching, Cover 계산 (while문 밖에서)
        self.build_equality_edges()
        matching, (Rc, Pc) = self.find_matching_and_cover(
            self.r_label, self.p_label
        )

        if self.debug:        
            print("=== Initial State ===")
            print(f"Initial equality edges: {self.Ey}")
            print(f"Initial matching: {matching}, size={len(matching)}")
            print(f"Initial vertex cover: Rc={list(self.Rc)}, Pc={list(self.Pc)}")
        
        # Perfect matching 초기 확인
        if len(matching) == n:
            assignment = np.full(n, -1, dtype=int)
            for (i, j) in matching:
                assignment[i] = j
            
            total_cost = float(sum(self.weights[i, assignment[i]] for i in range(n)))
            
            if self.debug:
                print(f"\nPerfect matching found immediately!")
                print(f"Assignment: {assignment}")
                print(f"Total cost: {total_cost}")
            
            return assignment, total_cost, (self.r_label, self.p_label)
        
        # while문 조건: perfect matching이 아니고 최대 반복 횟수 내에서 반복
        while len(matching) < n and iteration < max_iter:
            iteration += 1
            
            if self.debug:
                print(f"\n=== Iteration {iteration} ===")
                print(f"Current labels: r={self.r_label}, p={self.p_label}")
            
            # 2. step_1_a를 이용하여 E_cand 생성
            self.step_1_a()
            print(f"Candidate edges: {self.E_cand}") if self.debug else None
            
            # 각 candidate edge의 slack 값 출력
            for i, j in self.E_cand:
                slack = self.calculate_slack(i, j)
                print(f"  Edge ({i},{j}): slack = {slack}") if self.debug else None
            
            # 3. 생성된 E_cand를 이용하여 step_1_b 실행
            if len(self.E_cand) == 0:
                print("No candidate edges found!") if self.debug else None
                break
            
            delta = self.step_1_b()
            if self.debug:
                print(f"Delta: {delta}")
                print(f"Updated labels: r={self.r_label}, p={self.p_label}")
            
            if delta == 0:
                print("Delta is 0 - potential infinite loop!") if self.debug else None
                break
            
            # 4. update된 label을 이용하여 Ey 재생성하고, 재생성된 Ey를 기반으로 matching과 cover 생성
            self.build_equality_edges()
            matching, (Rc, Pc) = self.find_matching_and_cover(
                self.r_label, self.p_label
            )
            
            if self.debug:
                print(f"Updated equality edges: {self.Ey}")
                print(f"Updated matching: {matching}, size={len(matching)}")
                print(f"Updated vertex cover: Rc={list(self.Rc)}, Pc={list(self.Pc)}")
        
        # while문 종료 후 결과 처리
        if len(matching) == n:
            assignment = np.full(n, -1, dtype=int)
            for (i, j) in matching:
                assignment[i] = j
            
            total_cost = float(sum(self.weights[i, assignment[i]] for i in range(n)))
            
            if self.debug:
                print(f"\nPerfect matching found!")
                print(f"Assignment: {assignment}")
                print(f"Total cost: {total_cost}")
            
            return assignment, total_cost, (self.r_label, self.p_label)
        else:
            print(f"\nStopped after {iteration} iterations without perfect matching") if self.debug else None
            return None, None, None

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
        assignment, total_cost, _ = h.solve(max_iter=20)

        print(f"  → computed total cost: {total_cost}")
        print(f"  → expected optimal cost: {case['optimal_cost']}")
        if total_cost is not None and case['optimal_cost'] is not None:
            print(f"  → diff: {total_cost - case['optimal_cost']}")

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\n" + s.getvalue())
