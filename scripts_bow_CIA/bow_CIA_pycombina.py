import numpy as np
import matplotlib.pyplot as plt
import time
import pylab as pl
import pycombina
from pycombina import BinApprox, CombinaBnB

def bow_mapping(bow_array, dt, Num, dwell_time, stop_dwell_time):
    
    bow_left_array = np.abs(np.maximum(bow_array, 0))
    bow_right_array = np.abs(np.minimum(bow_array, 0)) 
    bow_off_array = 1 - (bow_right_array + bow_left_array)
    
    t = np.arange(Num+1) * dt
    b_rel = [[l, r, o] for l, r, o in zip(bow_left_array, bow_right_array, bow_off_array)]
    b_rel_T = list(map(list, zip(*b_rel)))
    # print(len(b_rel_T[1]))
    # print(t.size-1)


    binapprox = BinApprox(t, b_rel_T)

    assert(binapprox.n_c == 3)
    assert(binapprox.n_t == t.size-1)

    binapprox.set_min_up_times([dwell_time, dwell_time, stop_dwell_time])
            
    combina = CombinaBnB(binapprox)

    # combina.solve()
    combina.solve(verbosity=0)

    b_bin = binapprox.b_bin

    # print(len(b_bin[0]))

    # Solve
    # Extract solution
    new_left = b_bin[0,:]
    new_right = b_bin[1,:]
    new_bow_array = new_left - new_right
    return new_bow_array

if __name__ == "__main__":

    # bow_array = np.array([-1, -1, -1, -1, -1, -1, -1, 0.665, 1, 1, -1, -1, -1, -1, -1, -1, -0.41, 1.0, 1.0, 1.0])
    bow_array = np.array([-1, -1, -1, -0.8, -0.6, -0.4, 0.1, 0.0, -0.1, 0, -0.23, -0.4, -0.6, -0.8, -1, 0, 0.5, 0.8, 1.0, 1.0])
  

    dt = 0.5
    Num = len(bow_array)
    dwell_time = 2.0
    stop_dwell_time = 2.0
    t = time.time()
    new_bow = bow_mapping(bow_array, dt, Num, dwell_time, stop_dwell_time)
    elapsed = time.time() - t

    print(elapsed)
    print("Optimized bow array:", new_bow)

    # Plotting
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