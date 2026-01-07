import random
import math
import json
import numpy as np
from collections import deque

# Định nghĩa node

class Node:
    def __init__(self, ID, x, y):
        self.ID = ID
        self.x = x
        self.y = y
        self.neighbors = []
        self.parentID = None
        self.level = None


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


# BFS Tree

def build_bfs_tree(node_list, source=0):
    N = len(node_list)
    visited = [False] * N
    parent = [-1] * N

    q = deque([source])
    visited[source] = True

    while q:
        u = q.popleft()
        for v in node_list[u].neighbors:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                q.append(v)

    return all(visited), parent


def compute_levels(node_list, source=0):
    N = len(node_list)
    level = [None] * N
    level[source] = 0

    children = [[] for _ in range(N)]
    for i in range(N):
        p = node_list[i].parentID
        if p is not None:
            children[p].append(i)

    q = deque([source])
    while q:
        u = q.popleft()
        for v in children[u]:
            level[v] = level[u] + 1
            q.append(v)

    for i in range(N):
        node_list[i].level = level[i]

# Tạo topology hợp lệ

def generate_valid_topo(N=100, D=15, area=100, max_attempts=10000):

    center = area // 2

    for _ in range(max_attempts):
        nodes = [Node(0, center, center)]
        used = {(center, center)}

        for i in range(1, N):
            while True:
                x = random.randint(0, area)
                y = random.randint(0, area)
                if (x, y) not in used:
                    used.add((x, y))
                    nodes.append(Node(i, x, y))
                    break

        # Build neighbors
        for i in range(N):
            for j in range(i + 1, N):
                if distance(nodes[i], nodes[j]) <= D:
                    nodes[i].neighbors.append(j)
                    nodes[j].neighbors.append(i)

        ok, parent = build_bfs_tree(nodes)
        if ok:
            for i in range(N):
                nodes[i].parentID = None if parent[i] == -1 else parent[i]
            compute_levels(nodes)
            return nodes

    raise RuntimeError("Cannot generate valid topology")


# Lưu topology vào file JSON

def save_topologies(filename, num_topo=30, N=100, D=15):

    all_topos = []

    for t in range(num_topo):
        topo = generate_valid_topo(N=N, D=D)

        nodes_dict = {}
        for node in topo:
            nodes_dict[str(node.ID)] = {
                "x": node.x,
                "y": node.y,
                "parentID": node.parentID,
                "level": node.level
            }

        all_topos.append({
            "topo_id": t,
            "D": D,
            "nodes": nodes_dict
        })

        print(f"  Generated topo {t+1}/{num_topo}")

    with open(filename, "w") as f:
        json.dump({
            "N": N,
            "num_topologies": num_topo,
            "topologies": all_topos
        }, f, indent=2)

    print(f"[✓] Saved to {filename}")


# Load topology từ file JSON cho model

def load_topos_for_model(json_file, topo_id):

    with open(json_file, "r") as f:
        data = json.load(f)

    topo = data["topologies"][topo_id]
    nodes = topo["nodes"]
    D = topo["D"]

    node_ids = sorted([int(k) for k in nodes.keys()])
    N = len(node_ids)

    # positions
    pos = {i: (nodes[str(i)]["x"], nodes[str(i)]["y"]) for i in node_ids}

    # adjacency matrix
    A = np.zeros((N, N), dtype=np.float32)

    for i in node_ids:
        xi, yi = pos[i]
        for j in node_ids:
            if i == j:
                continue
            xj, yj = pos[j]
            if math.hypot(xi - xj, yi - yj) <= D:
                A[i, j] = 1.0
                A[j, i] = 1.0

    parent = [nodes[str(i)]["parentID"] for i in node_ids]
    level = [nodes[str(i)]["level"] for i in node_ids]

    return {
        "N": N,
        "A": A,
        "parent": parent,
        "level": level
    }


# Tạo và lưu nhiều topology với các cấu hình khác nhau

if __name__ == "__main__":

    D_list = [15, 30, 45]
    N_list = [100, 150, 300, 500, 1000]
    NUM_TOPO = 30

    for D in D_list:
        for N in N_list:
            print(f"\n=== Configuration D={D}, N={N} ===")

            filename = f"topologies_D{D}_N{N}.json"
            save_topologies(
                filename=filename,
                num_topo=NUM_TOPO,
                N=N,
                D=D
            )
