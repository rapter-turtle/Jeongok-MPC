import numpy as np
from gurobipy import Model, GRB, QuadExpr
import matplotlib.pyplot as plt
import time

def bow_mapping(model, bow_array, bow_switch, dt, Num, dwell_time, stop_dwell_time):
    bow_left_array = np.abs(np.maximum(bow_array, 0))
    bow_right_array = np.abs(np.minimum(bow_array, 0)) 

    # print(bow_left_array)
    # print(bow_right_array)

    total_variables = 2 * Num + 1
    eps = 0.0
    dwell_count = int(dwell_time / dt)
    stop_dwell_count = int(stop_dwell_time / dt)

    
    model.setParam('OutputFlag', 0)

    # Variables
    x = []
    x.append(model.addVar(lb=-50, ub=50, vtype=GRB.CONTINUOUS, name="x0"))  # offset
    for i in range(2 * Num):
        x.append(model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x{i + 1}"))

    # Objective: Minimize offset (x[0])
    obj = QuadExpr()
    obj += x[0]
    model.setObjective(obj, GRB.MINIMIZE)

    # Mode switch constraints
    for i in range(Num):
        model.addConstr(x[2 * i + 1] + x[2 * i + 2] <= 1.1)

    # CIA constraints (cumulative error bounds)
    for i in range(1, Num+1):
        bow_right_sum = np.sum(bow_right_array[:i])
        bow_left_sum = np.sum(bow_left_array[:i])

        pattern_left = [1 if j % 2 == 0 else 0 for j in range(2 * i)]
        pattern_right = [0 if j % 2 == 0 else 1 for j in range(2 * i)]

        lhs_left = dt * sum(pattern_left[j] * x[j + 1] for j in range(2 * i))
        lhs_right = dt * sum(pattern_right[j] * x[j + 1] for j in range(2 * i))

        model.addConstr(x[0] + lhs_left >= dt * bow_left_sum)
        model.addConstr(x[0] - lhs_left >= -dt * bow_left_sum)
        model.addConstr(x[0] + lhs_right >= dt * bow_right_sum)
        model.addConstr(x[0] - lhs_right >= -dt * bow_right_sum)

    if bow_switch ==0:
        for j in range(1, dwell_count):
            model.addConstr(bow_switch - x[1] + x[2 * j + 1] >= 0)
            model.addConstr(bow_switch - x[2] + x[2 * j] >= 0)

    elif bow_switch ==1:
        for j in range(1, dwell_count):
            model.addConstr(- x[2] + x[2 * j] >= 0)

    elif bow_switch ==-1:
        for j in range(1, dwell_count):
            model.addConstr(- x[1] + x[2 * j + 1] >= 0)


    # # Dwell ON constraints
    for i in range(1, Num - 1):
        idx = 2 * i - 1
        # print(idx)
        model.addConstr(x[idx] - x[idx + 2] + x[idx + 4] >= 0)
        model.addConstr(x[idx + 1] - x[idx + 3] + x[idx + 5] >= 0)
        # print(idx + 5, Num)
        for j in range(0, dwell_count):
            if idx + 2 * j + 6 < total_variables:
                model.addConstr(x[idx] - x[idx + 2] + x[idx + 2 * j + 6] >= 0)
                model.addConstr(x[idx + 1] - x[idx + 3] + x[idx + 2 * j + 7] >= 0)
                # print(idx + 2 * j + 7, Num)


    # # Dwell OFF constraints
    if bow_switch == 1:
        for j in range(0, stop_dwell_count):
            model.addConstr(x[2 * j + 2] <= 0)

    if bow_switch == -1:
        for j in range(0, stop_dwell_count):
            model.addConstr(x[2 * j + 1] <= 0)  


    for i in range(1, Num - 1):
        idx = 2 * i - 1
        model.addConstr(x[idx] - x[idx + 2] + x[idx + 4] <= 1)
        model.addConstr(x[idx + 1] - x[idx + 3] + x[idx + 5] <= 1)
        for j in range(0, stop_dwell_count):
            if idx + 2 * j + 6 < total_variables:
                model.addConstr(x[idx] - x[idx + 2] + x[idx + 2 * j + 6] <= 1)
                model.addConstr(x[idx] - x[idx + 2] + x[idx + 2 * j + 7] <= 1)
                model.addConstr(x[idx + 1] - x[idx + 3] + x[idx + 2 * j + 6] <= 1)
                model.addConstr(x[idx + 1] - x[idx + 3] + x[idx + 2 * j + 7] <= 1)

    # Solve
    model.optimize()
    result = [v.X for v in x]
    # print(result)
    # Extract solution
    new_left = np.array(result[1:2 * Num + 1:2])
    new_right = np.array(result[2:2 * Num + 1:2])
    new_bow_array = new_left - new_right
    return new_bow_array

if __name__ == "__main__":
    # 예제 입력
    # bow_array = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -0.11, 1, 1, 1, 0.98, 0.43, 0.24, 0.4])
    bow_array = np.array([-1, -1, -1, -1, -1, -1, -1, 0.665, 1, 1, -1, -1, -1, -1, -1, -1, -0.41, 1.0, 1.0, 1.0])

    # bow_array = np.array([-0.17338943, -0.29207375, -0.38623622, -0.44877279, -0.47656849, -0.46931089, -0.42858701, -0.35774312, -0.26214989, -0.14965445, -0.03051529, 0.08356666, 0.18038598, 0.24928092, 0.28420294, 0.28259033, 0.24358795, 0.16734687, 0.05490006, -0.09167268])

# -1.         -1.         -1.         -1.         -1.         -1.
#  -1.          0.66544232  1.          1.         -1.         -1.
#  -1.         -1.         -1.         -1.         -0.40753806  1.
#   1.          1.    
    dt = 0.5
    Num = len(bow_array)
    dwell_time = 3.0
    stop_dwell_time = 5.0
    model = Model("bow_mapping")
    t = time.time()
    new_bow = bow_mapping(model,bow_array, dt, Num, dwell_time, stop_dwell_time)
    elapsed = time.time() - t

    print(elapsed)
    # 결과 출력
    # print("Original bow array:", bow_array)
    print("Optimized bow array:", new_bow)

    # 플로팅
    time = np.arange(Num) * dt
    plt.figure(figsize=(8, 4))
    plt.plot(time, bow_array, label='Original Bow', marker='o')
    plt.plot(time, new_bow, label='Optimized Bow', marker='x')
    plt.title("Bow Mapping Result")
    plt.xlabel("Time [s]")
    plt.ylabel("Bow Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()