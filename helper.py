import numpy as np
import matplotlib.pyplot as plt

def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)

def plot_arr(arr, label = None, window_size = 101):
    def rollavg_cumsum(a, n):
        assert n % 2 == 1
        cumsum_vec = np.cumsum(np.insert(a, 0, 0))
        return (cumsum_vec[n:] - cumsum_vec[:-n]) / n

    arr = np.array(arr)

    moving_avg = rollavg_cumsum(arr, window_size)
    plt.plot(np.arange(len(moving_avg)), moving_avg)
    plt.title(label)
    plt.pause(0.001)
