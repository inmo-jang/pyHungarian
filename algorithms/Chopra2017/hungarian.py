import numpy as np
import random

class Hungarian:
    def __init__(self):
        weights = self.create_weight_matrix()
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

    def create_weight_matrix(self):
        weight_matrix = np.array([
            [5, np.inf, np.inf, np.inf, np.inf],
            [3, 7, 3, np.inf, np.inf],
            [6, np.inf, 2, np.inf, 4],
            [np.inf, np.inf, 5, 6, 2],
            [np.inf, 4, np.inf, 1, 6]
        ], dtype=float)
        return weight_matrix

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
        adj = [[] for _ in range(5)]
        for r_idx, p_idx in self.Ey:
            adj[r_idx].append(p_idx)
        
        # Initialize matching arrays
        match_r = [-1] * 5  # match_r[r] = p means row r is matched to column p
        match_p = [-1] * 5  # match_p[p] = r means column p is matched to row r
        
        # Random row ordering for diversity
        rows = list(range(5))
        random.shuffle(rows)
        
        # Find maximum matching using randomized row order
        for r_idx in rows:
            visited = [False] * 5
            self.bmp(r_idx, match_r, match_p, adj, visited)
        
        # Extract matching pairs
        matching = []
        for r_idx in range(5):
            if match_r[r_idx] != -1:
                matching.append((r_idx, match_r[r_idx]))
        
        # Calculate vertex cover using König's theorem
        # Randomly choose between standard and alternative methods
        vertex_cover_method = random.choice(['standard', 'alternative'])
        
        if vertex_cover_method == 'standard':
            # Standard: Start from unmatched rows, find alternating paths
            unmatched_r = [r for r in range(5) if match_r[r] == -1]
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
            Rc = [r for r in range(5) if r not in reachable_r]
            Pc = list(reachable_p)
            
        else:  # alternative
            # Alternative: Start from unmatched columns, find alternating paths
            unmatched_p = [p for p in range(5) if match_p[p] == -1]
            reachable_r = set()
            reachable_p = set()
            
            def dfs_from_unmatched_cols(p):
                if p in reachable_p:
                    return
                reachable_p.add(p)
                for r in range(5):
                    if p in adj[r] and r not in reachable_r:
                        reachable_r.add(r)
                        if match_r[r] != -1:
                            dfs_from_unmatched_cols(match_r[r])
            
            for p in unmatched_p:
                dfs_from_unmatched_cols(p)
            
            # Vertex cover: reachable rows + unreachable columns
            Rc = list(reachable_r)
            Pc = [p for p in range(5) if p not in reachable_p]
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

    def solve_debug(self, max_iter=5):
        """
        Hungarian algorithm debug version with limited iterations
        """
        n = self.r
        iteration = 0
        
        print("=== Hungarian Algorithm Debug (While Loop) ===")
        print(f"Initial labels: r={self.r_label}, p={self.p_label}")
        print()
        
        # 초기 Ey, Matching, Cover 계산 (while문 밖에서)
        self.build_equality_edges()
        matching, (Rc, Pc) = self.find_matching_and_cover(
            self.r_label, self.p_label
        )
        
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
            
            print(f"\nPerfect matching found immediately!")
            print(f"Assignment: {assignment}")
            print(f"Total cost: {total_cost}")
            
            return assignment, total_cost, (self.r_label, self.p_label)
        
        # while문 조건: perfect matching이 아니고 최대 반복 횟수 내에서 반복
        while len(matching) < n and iteration < max_iter:
            iteration += 1
            
            print(f"\n=== Iteration {iteration} ===")
            print(f"Current labels: r={self.r_label}, p={self.p_label}")
            
            # 2. step_1_a를 이용하여 E_cand 생성
            self.step_1_a()
            print(f"Candidate edges: {self.E_cand}")
            
            # 각 candidate edge의 slack 값 출력
            for i, j in self.E_cand:
                slack = self.calculate_slack(i, j)
                print(f"  Edge ({i},{j}): slack = {slack}")
            
            # 3. 생성된 E_cand를 이용하여 step_1_b 실행
            if len(self.E_cand) == 0:
                print("No candidate edges found!")
                break
            
            delta = self.step_1_b()
            print(f"Delta: {delta}")
            print(f"Updated labels: r={self.r_label}, p={self.p_label}")
            
            if delta == 0:
                print("Delta is 0 - potential infinite loop!")
                break
            
            # 4. update된 label을 이용하여 Ey 재생성하고, 재생성된 Ey를 기반으로 matching과 cover 생성
            self.build_equality_edges()
            matching, (Rc, Pc) = self.find_matching_and_cover(
                self.r_label, self.p_label
            )
            
            print(f"Updated equality edges: {self.Ey}")
            print(f"Updated matching: {matching}, size={len(matching)}")
            print(f"Updated vertex cover: Rc={list(self.Rc)}, Pc={list(self.Pc)}")
        
        # while문 종료 후 결과 처리
        if len(matching) == n:
            assignment = np.full(n, -1, dtype=int)
            for (i, j) in matching:
                assignment[i] = j
            
            total_cost = float(sum(self.weights[i, assignment[i]] for i in range(n)))
            
            print(f"\nPerfect matching found!")
            print(f"Assignment: {assignment}")
            print(f"Total cost: {total_cost}")
            
            return assignment, total_cost, (self.r_label, self.p_label)
        else:
            print(f"\nStopped after {iteration} iterations without perfect matching")
            return None, None, None

# Test
if __name__ == "__main__":
    h = Hungarian()
    print("Weight matrix:")
    print(h.weights)
    print()
    
    h.solve_debug(max_iter=10)
