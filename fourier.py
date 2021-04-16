import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 2 # period

def f(x): # one period of the periodic function
    if -1 <= x < 0:
        return 2 * x + 2
    elif 0 <= x < 0.5:
        return -2
    elif 0.5 <= x <= 1:
        return 1
    else:
        return 0

x = np.linspace(-2, 2, 1000)
y = [f(xi) for xi in x]

plt.figure()
plt.plot(x, y)
plt.grid()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("One period of the signal")

# Fourier coefficients
a_0 = 2 / T * quad(f, -T / 2, T / 2)[0]

def a(n):
    return (4 * np.sin(np.pi * n / 2) ** 2 / (np.pi * n) -
            2 * np.sin(np.pi * n / 2) +
            np.sin(np.pi * n) - np.sin(np.pi * n / 2)) / (np.pi * n)

def b(n):
    return (2 * (np.sin(np.pi * n) - np.pi * n) / (np.pi * n) -
            np.cos(np.pi * n) + np.cos(np.pi * n / 2) +
            4 * (np.cos(np.pi * n / 4) ** 2 - 1)) / (np.pi * n)

def amplitude(n):
    if n == 0:
        return a_0
    return np.sqrt(a(n) ** 2 + b(n) ** 2)

def phase(n):
    if n == 0:
        return 0
    return np.angle(a(n) - 1j*b(n))

def frequency(n):
    return n / T

def harmonics(x, n):
    return amplitude(n) * np.cos(2 * np.pi * frequency(n) * x + phase(n))

n = np.arange(-50, 50)
amp = [amplitude(ni) for ni in n]
ph = [phase(ni) for ni in n]

# spectrum of amplitudes
plt.figure()
plt.stem(n, amp)
plt.grid()
plt.xlabel("$n$")
plt.ylabel("$d(n)$")

# spectrum of [phases]
plt.figure()
plt.stem(n, ph)
plt.grid()
plt.xlabel("$n$")
plt.ylabel("$\\theta(n)$, rad")

def s(x, number=3):
    result = 0
    for ni in range(1, number + 1):
        result += harmonics(x, ni)
    return a_0 + result

x = np.linspace(-2 * T, 2 * T, 1000)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# Fourier approximations of the signal f(x)
for ax, n in zip((ax1, ax2, ax3, ax4), (3, 10, 20,100)):
    ax.plot(x, s(x, n))
    ax.set_title("$S_{{{}}}$".format(n))
    ax.grid()
    ax.set_xlabel("$x$")

plt.show()