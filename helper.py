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
    def moving_avg(a, n):
        new_a = []
        for i in range(len(a)):
            if i < n:
                new_a.append(0)
            else:
                new_a.append(np.mean(a[i-window_size:i]))
        return new_a

    arr = np.array(arr)
    if window_size > 1:
        arr = moving_avg(arr, window_size)
    plt.plot(np.arange(len(arr)), arr)
    plt.title(label)
    plt.pause(0.001)




