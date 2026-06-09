import math

def get_distance(city1, city2):
    # 算兩點之間的直線距離（歐幾里得距離）
    return math.dist(city1, city2)

def height(solution, cities):
    """
    目標函數：計算目前的總距離，然後加個負號
    （因為爬山演算法是往高處爬，所以距離越短、負越多，Height 值就會越大）
    """
    total_distance = 0
    n = len(solution)
    for i in range(n):
        city_a = cities[solution[i]]
        # 確保最後一個城市會連回第一個城市，繞成一個圈
        city_b = cities[solution[(i + 1) % n]] 
        total_distance += get_distance(city_a, city_b)
    return -total_distance

def get_best_neighbor(solution, cities):
    """
    找鄰居：用 2-opt 演算法來點新花樣
    概念是隨機挑兩條路打斷，然後交叉反轉重新連起來，看會不會比較快
    """
    best_neighbor = solution[:]
    best_h = height(solution, cities)
    n = len(solution)

    # 暴力窮舉所有可能的兩條邊組合，來產生各種鄰居解
    for i in range(n - 1):
        for j in range(i + 2, n):
            # 如果是頭尾相連的同一條邊，就不用浪費時間換了（換了等於沒換）
            if i == 0 and j == n - 1:
                continue
            
            # 用 Python 切片把 i+1 到 j 這段路徑反轉，這樣就完成邊的交換了
            neighbor = solution[:i+1] + solution[i+1:j+1][::-1] + solution[j+1:]
            
            # 看看這個新鄰居有沒有比較厲害
            current_h = height(neighbor, cities)
            if current_h > best_h:
                best_h = current_h
                best_neighbor = neighbor

    return best_neighbor, best_h

def hill_climbing(cities):
    """ 爬山演算法的主流程 """
    # 隨便給個初始解，就先照著城市編號 1 -> 2 -> 3 ... 這樣走
    current_solution = list(cities.keys())
    current_h = height(current_solution, cities)
    
    while True:
        # 從所有鄰居裡面，挑一個 Height 最高（也就是距離最短）的
        best_neighbor, neighbor_h = get_best_neighbor(current_solution, cities)
        
        # 如果四周的鄰居都沒有比現在的位置好，代表走到極值（局部最佳解）了，直接收工
        if neighbor_h <= current_h:
            break
            
        # 如果鄰居比較好，就往那個方向移過去
        current_solution = best_neighbor
        current_h = neighbor_h

    # 回傳最終的路徑，順便把 Height 轉回正數的實際距離
    return current_solution, -current_h

if __name__ == "__main__":
    # 測試資料：城市編號 1~n，後面是它們的 (X, Y) 座標
    cities_data = {
        1: (0, 0),
        2: (2, 8),
        3: (5, 2),
        4: (7, 6),
        5: (8, 1),
        6: (3, 5)
    }
    
    print("開始執行爬山演算法...")
    best_path, min_distance = hill_climbing(cities_data)
    
    print("\n=== 執行結果 ===")
    print(f"最佳路徑 (最終解): {best_path}")
    print(f"最短總距離: {min_distance:.2f}")