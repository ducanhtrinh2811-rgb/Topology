import os
import random
import math
from collections import deque

# Định nghĩa Node
class Node:
    def __init__(self, ID, x, y):
        self.ID = ID
        self.x = x
        self.y = y
        self.neighbors = []

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# Kiểm tra kết nối bằng BFS
def is_connected(nodes, source=0):
    N = len(nodes)
    visited = [False] * N
    q = deque([source])
    visited[source] = True
    
    while q:
        u = q.popleft()
        for v in nodes[u].neighbors:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    
    return all(visited)

# Xây dựng danh sách neighbors

def build_neighbors(nodes, D):

    for node in nodes:
        node.neighbors = []
    
    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if distance(nodes[i], nodes[j]) <= D:
                nodes[i].neighbors.append(j)
                nodes[j].neighbors.append(i)

# Tạo topology
def generate_valid_topo(area, D, p, max_attempts=10000):

    # Tính N theo công thức: N = (ρ × L²) / π

    N = int((p * area * area) / math.pi)
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

        build_neighbors(nodes, D)

        if is_connected(nodes):
            return nodes, N

    raise RuntimeError(
        f"Cannot generate connected topology "
        f"(D={D}, p={p}, area={area}, N={N})"
    )

# Lưu topology vào file TXT
def save_topo_txt(nodes, filename, N, D, area):
    try:
        with open(filename, "w") as f:
            f.write(f"# N={N}  D={D}  Area={area}\n")
            f.write("# id x y\n")
            
            # Dữ liệu nodes
            for n in nodes:
                f.write(f"{n.ID} {n.x} {n.y}\n")
        return True
    except Exception as e:
        print(f"Không thể lưu file {filename}: {e}")
        return False

# Load topology từ file TXT
def load_topo_txt(filename):
    nodes = []
    metadata = {}
    
    try:
        with open(filename) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:
                    continue
                
                if line.startswith("# N="):
                    try:
                        parts = line[2:].split()
                        for part in parts:
                            key, val = part.split("=")
                            metadata[key] = int(val)
                    except:
                        pass
                    continue
                
                if line.startswith("#"):
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) != 3:
                        print(f"Dòng {line_num} format sai, bỏ qua: {line}")
                        continue
                    
                    node_id, x, y = parts
                    nodes.append(Node(int(node_id), float(x), float(y)))
                    
                except ValueError as e:
                    print(f"Dòng {line_num} không parse được, bỏ qua: {line}")
                    continue
        
        if 'D' in metadata and nodes:
            build_neighbors(nodes, metadata['D'])
            print(f"Đã rebuild neighbors với D={metadata['D']}")
        
        return nodes, metadata
        
    except FileNotFoundError:
        print(f"Không tìm thấy file: {filename}")
        return [], {}
    except Exception as e:
        print(f"Lỗi khi đọc file {filename}: {e}")
        return [], {}

# Tạo thư mục nếu chưa có
def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Không thể tạo thư mục {path}: {e}")
        return False

# Tạo database topology
def generate_topology_database(
    root_dir="TopologyDB",
    D_list=(15, 30, 45),
    p_list=(0.01, 0.015, 0.03, 0.05, 0.1),
    num_topo=30,
    area=100
):
    # Tính tổng số file cần tạo
    total_files = len(D_list) * len(p_list) * num_topo
    current_file = 0
    
    print(f"Bắt đầu tạo DATABASE TOPOLOGY")
    print(f"Tổng số file cần tạo: {total_files}")
    print(f"Cấu hình: D={D_list}, p={p_list}, num_topo={num_topo}")
    
    success_count = 0
    fail_count = 0
    
    for D in D_list:
        for p in p_list:
            # Tính N từ công thức
            N = int((p * area * area) / math.pi)
            print(f"\nĐang tạo: D={D}, ρ={p} → N={N}")
            
            # Tạo thư mục
            folder = os.path.join(root_dir, f"D{D}", f"N{N}")
            if not ensure_dir(folder):
                print(f"Bỏ qua D={D}, N={N} do không tạo được thư mục")
                fail_count += num_topo
                continue
            
            # Tạo từng topology
            for k in range(num_topo):
                current_file += 1
                progress = (current_file / total_files) * 100
                
                try:
                    nodes, N_actual = generate_valid_topo(area=area, D=D, p=p)
                    
                    # Lưu file
                    filename = os.path.join(folder, f"topo_D{D}_N{N_actual}_{k+1:03d}.txt")
                    if save_topo_txt(nodes, filename, N_actual, D, area):
                        print(f"Saved")
                        success_count += 1
                    else:
                        print(f"Failed to save")
                        fail_count += 1
                        
                except RuntimeError as e:
                    print(f"{e}")
                    fail_count += 1
                except Exception as e:
                    print(f"Error: {e}")
                    fail_count += 1
    
    print(f"Kết quả tạo DATABASE")
    print(f"Thành công: {success_count}/{total_files}")
    print(f"Thất bại:   {fail_count}/{total_files}")

# Entry point
if __name__ == "__main__":
    # Cấu hình
    D_list = [15, 30, 45]
    
    # Tính ρ từ N theo công thức: ρ = (N × π) / L²
    AREA = 100
    N_desired = [100, 150, 300, 500, 1000]
    
    # Tính mật độ ρ tương ứng
    p_list = [(N * math.pi) / (AREA * AREA) for N in N_desired]
    
    NUM_TOPO = 30
    
    # Hiển thị các giá trị N và ρ tương ứng
    print("Giá trị ρ tương ứng với N mong muốn:")
    for N, p in zip(N_desired, p_list):
        N_actual = int((p * AREA * AREA) / math.pi)
        print(f"  N = {N} → ρ = {p:.6f} → N_actual = {N_actual}")
    print()
    
    # Tạo database
    generate_topology_database(
        root_dir="TopologyDB",
        D_list=D_list,
        p_list=p_list,
        num_topo=NUM_TOPO,
        area=AREA
    )