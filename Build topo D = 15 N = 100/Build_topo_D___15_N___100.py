#BUILDING TOPOLOGY

import torch
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import json 

class Node:
    def __init__(self, ID, x, y):
        self.ID = ID
        self.x = x
        self.y = y
        self.neighbors = []
        self.parentID = None
        self.level = None
        self.parent_conflict_free = None
        self.level_conflict_free = None


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

from collections import deque
import random

def build_bfs_tree(node_list, source=0):
    N = len(node_list)

    visited = [False] * N
    parent  = [-1] * N

    q = deque([source])
    visited[source] = True

    while q:
        u = q.popleft()
        for v in node_list[u].neighbors:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                q.append(v)

    is_tree = all(visited)
    return is_tree, parent


def compute_levels_from_parent(node_list, source=0):

    N = len(node_list)
    level = [None] * N
    level[source] = 0

    # Tạo danh sách con cho mỗi node dựa trên parentID (tránh duyệt neighbors)
    children = [[] for _ in range(N)]
    for i in range(N):
        p = node_list[i].parentID
        if p is not None:
            children[p].append(i)

    q = deque([source])
    while q:
        u = q.popleft()
        for v in children[u]:
            if level[v] is None:
                level[v] = level[u] + 1
                q.append(v)

    # Gán vào object Node
    for i in range(N):
        node_list[i].level = level[i]


def generate_valid_topo(num_node=150, x_range=100, y_range=100, comm_range=20, max_attempts=10000):

    center_x = x_range // 2
    center_y = y_range // 2

    for attempt in range(max_attempts):
        node_list = []
        node_list.append(Node(0, center_x, center_y))

        # Dùng set để chống trùng tọa độ nhanh hơn
        used_pos = {(center_x, center_y)}
        for i in range(1, num_node):
            while True:
                x = random.randint(0, x_range)
                y = random.randint(0, y_range)
                if (x, y) not in used_pos:
                    used_pos.add((x, y))
                    node_list.append(Node(i, x, y))
                    break

        # Reset neighbors (phòng trường hợp Node tái dùng ở đâu đó)
        for node in node_list:
            node.neighbors = []

        # Xây neighbors theo comm_range
        for i in range(num_node):
            for j in range(i + 1, num_node):
                if distance(node_list[i], node_list[j]) <= comm_range:
                    node_list[i].neighbors.append(j)
                    node_list[j].neighbors.append(i)

        # BFS check
        is_tree, parent = build_bfs_tree(node_list, source=0)
        if is_tree:
            # Gán parentID
            for i in range(num_node):
                node_list[i].parentID = None if parent[i] == -1 else parent[i]

            # Tính level
            compute_levels_from_parent(node_list, source=0)

            print(f"[✓] BFS hợp lệ sau {attempt + 1} lần thử")
            return node_list

    raise RuntimeError("Không tìm được topo hợp lệ sau quá nhiều lần thử.")

# === THỰC THI LẠI SAU KHI SỬA HÀM ===
node_list = generate_valid_topo()

# Hiển thị thông tin node (kiểm tra vị trí node 0)
for node in node_list:
    print(f"Node {node.ID}: (x={node.x}, y={node.y}), neighbors={node.neighbors}, parent={node.parentID}, level={node.level}")

# Vẽ đồ thị (sẽ thấy node 0 ở trung tâm)
G = nx.Graph()
for node in node_list:
    G.add_node(node.ID, pos=(node.x, node.y))
    for neighbor in node.neighbors:
        G.add_edge(node.ID, neighbor)

pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Topo always-on 50 node (có cây BFS, node 0 ở trung tâm)")
plt.show()

# Generate multiple topos

def generate_multiple_topos(
    num_topo=30,
    num_node=100,      # N = 100
    comm_range=15,     # D = 15
    x_range=100,
    y_range=100
):
    topo_list = []

    for k in range(num_topo):
        topo = generate_valid_topo(
            num_node=num_node,
            x_range=x_range,
            y_range=y_range,
            comm_range=comm_range
        )
        topo_list.append(topo)
        print(f"Topo {k+1}/{num_topo} generated")

    return topo_list

# Run experiment: D=15, N=100

D = 15
N = 100
NUM_TOPO = 30

all_topos = []

for i in range(NUM_TOPO):
    print(f"\n=== Generating topology {i+1}/{NUM_TOPO} ===")
    topo = generate_valid_topo(
        num_node=N,
        x_range=100,
        y_range=100,
        comm_range=D 
    )
    all_topos.append(topo)

print(f"\nGenerated {len(all_topos)} topologies with D={D}, N={N}")

# Save to JSON
def topo_to_dict(topo):
    topo_dict = {}
    for node in topo:
        topo_dict[node.ID] = {
            'x': node.x,
            'y': node.y,
            'neighbors': node.neighbors,
            'parentID': node.parentID,
            'level': node.level
        }
    return topo_dict

json_data = {
    "D": D,
    "N": N,
    "num_topologies": len(all_topos),
    "topologies": [
        {
            "topo_id": i,
            **topo_to_dict(topo)
        }
        for i, topo in enumerate(all_topos)
    ]
}

with open("topologies_D15_N100.json", "w") as f:
    json.dump(json_data, f, indent=2)

print("Saved to topologies_D15_N100.json")

# Load Topologies for Model
def load_topos_for_model(json_file, topo_id):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    topo = data["topologies"][topo_id]
    node_ids = sorted([k for k in topo.keys() if k != "topo_id"])
    N = len(node_ids)

    # Adjacency matrix A

    A = np.zeros((N, N), dtype=np.float32)

    for i in node_ids:
        i = int(i)
        for j in topo[str(i)]["neighbors"]:
            j = int(j)
            A[i, j] = 1.0
            A[j, i] = 1.0   

    # Tree Structure
    
    children = [[] for _ in range(N)]
    for i in node_ids:
        p = topo[i]["parentID"]
        if p is not None:
            children[p].append(int(i))

    def count_descendants(u):
        return sum(1 + count_descendants(c) for c in children[u])


    # Node features X
    X = np.zeros((N, 6), dtype=np.float32)

    for i in node_ids:
        i = int(i)
        num_children = len(children[i])
        is_leaf = 1 if num_children == 0 else 0
        is_root = 1 if topo[str(i)]["parentID"] is None else 0

        X[i] = [
            i,                          # x1: ID
            num_children,               # x2: children count
            is_leaf,                    # x3: leaf
            topo[str(i)]["level"],      # x4: hop distance
            is_root,                    # x5: root
            count_descendants(i)        # x6: descendants
        ]

    return A, X


    
