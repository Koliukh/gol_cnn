import h5py
import numpy as np
import tqdm
from gol_conf import *

class GameOfLife:
    @staticmethod
    def gen_random_state(width: int, height: int):
        state = (np.random.rand(width, height) * 2).astype(np.int32)
        return state

    @staticmethod
    def next_state(state):
        l_neib = np.roll(state, 1, 0)
        r_neib = np.roll(state, -1, 0)
        u_neib = np.roll(state, 1, 1)
        d_neib = np.roll(state, -1, 1)
        ul_neib = np.roll(l_neib, 1, 1)
        dl_neib = np.roll(l_neib, -1, 1)
        ur_neib = np.roll(r_neib, 1, 1)
        dr_neib = np.roll(r_neib, -1, 1)

        neibs = l_neib + r_neib + u_neib + d_neib + ul_neib + dl_neib + ur_neib + dr_neib
        next_state = np.copy(state)
        next_state[(neibs < 2) | (neibs > 3)] = 0
        next_state[neibs == 3] = 1

        return next_state


dataset_name="dataset_test_"+str(width)+"x"+str(height)+"x"+str(n_samples_test)+".h5"

try:
    data_file = h5py.File(dataset_name, 'r')
    x_test = data_file["x_test"][:]
    y_test = data_file["y_test"][:]
    data_file.close()
except OSError:
    print("Generate x_test")
    x_test = []
    for _ in tqdm.trange(n_samples_test):
        x_test.append(GameOfLife.gen_random_state(width, height))

    x_test = np.array(x_test)

    print("Generate y_test")
    y_test = np.zeros_like(x_test)
    for i, x in tqdm.tqdm(enumerate(x_test), total=len(x_test)):
        y_test[i] = GameOfLife.next_state(x)

    data_file = h5py.File(dataset_name, 'w')
    data_file.create_dataset("x_test", data=x_test)
    data_file.create_dataset("y_test", data=y_test)
    data_file.close()

print("Dataset shape: {x_test.shape}")
