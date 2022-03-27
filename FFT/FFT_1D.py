#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

warnings.filterwarnings('ignore')

from numpy.fft import fft, ifft, fftshift, hfft, ihfft, rfft, irfft
RES = 1000
TOT_FRAME = 100

signal = np.zeros((RES,))
signal[2*(RES//5):3*(RES//5)] = 1
plt.style.use('seaborn-pastel')


fig, axes = plt.subplots(nrows=4, ncols=1)

def animate_low_high(i):
    # signal
    curr_space = np.abs(np.geomspace(RES//2, 1, num=TOT_FRAME)-RES//2)
    # curr_space = np.geomspace(1, RES//2, num=TOT_FRAME)
    i = curr_space[i].astype(int)
    print(i)

    x = np.arange(len(signal))
    for ax in axes:
        ax.clear()
    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_xlim(0, RES)
    axes[3].set_ylim(-0.5, 1.5)
    axes[3].set_xlim(0, RES)
    axes[1].set_ylim(-200, 200)
    axes[1].set_xlim(0, 500)
    axes[2].set_ylim(-2, 2)
    axes[2].set_xlim(0, 500)

    axes[0].plot(x, signal)

    freq = rfft(signal, RES).astype(float)
    sorted_ind = np.arange(len(freq))[::-1]
    # sorted_ind = np.arange(len(freq))
    for count, ind in enumerate(sorted_ind):
        ind = int(ind)
        if count > i:
            freq[ind] = 0.
    axes[1].plot(x[:501], freq)

    mult = sorted_ind[i-1]
    tmp_freq = np.zeros((RES//2 + 1,))
    tmp_freq[sorted_ind[i-1]] = freq[sorted_ind[i-1]]
    tmp_reconst = irfft(tmp_freq, RES//2)
    axes[2].plot(x[:500], tmp_reconst)

    reconst = irfft(freq, RES)
    axes[3].plot(x, reconst)
    return axes

anim = FuncAnimation(fig, animate_low_high, frames=TOT_FRAME)

anim.save('1D_rev.gif', writer='imagemagick', dpi=125)