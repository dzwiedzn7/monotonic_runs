import numpy as np
from scipy.signal import sawtooth

import matplotlib.pyplot as plt

# Generate a sample AR(1) time series
#np.random.seed(0)
n = 1000  # Number of data points
phi = 0.1  # AR parameter
sigma = 0.1  # Standard deviation of noise
scale = 3
threshold = np.random.normal(scale=sigma)
a = 0.5
b = 3
n_terms = 50 # Number of terms in the serie for Weierstrass
df = 2 # The degrees of freedom of the chi-square distribution.

def autoregressive_function(n,phi,sigma,scale):
    y = np.zeros(n)
    y[0] = np.random.normal()
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.normal(scale=sigma*scale)
    return y

def asymmetric_autoregressive_jp(n,phi,sigma,scale):
    y = np.zeros(n)
    y[0] = np.random.normal()

    for t in range(1, n):
        threshold = np.random.normal(scale=sigma)
        if y[t-1] < threshold:
            y[t] = phi * y[t-1] + np.random.normal(scale=sigma*scale)
        else:
            y[t] = phi * y[t-1] + abs(np.random.normal(scale=sigma*scale))
    return y

def asymmetric_weierstrass_sawtooth( a, b, n):
    """
    Compute an asymmetric Weierstrass-like function using sawtooth waves.

    Parameters:
    x: array-like, input values
    a: float, scaling factor (0 < a < 1)
    b: int, frequency factor (odd integer, ab > 1 + 3π/2)
    n_terms: int, number of terms to sum

    Returns:
    array-like, function values
    """
    x = np.linspace(-2, 2, n) + np.random.normal(0, 0.1, n)
    x = np.asarray(x)
    return sum(a ** n * sawtooth(b ** n * np.pi * x) for n in range(50))

def weierstrass(l, D):
    print(l)
    x = np.concatenate((-np.arange(-l, 0) / l, np.arange(0, l + 1) / l)).astype(np.float64)
    w = np.zeros_like(x, dtype=np.float64)
    for i in range(101):
        part1 = 2 ** (-i * (2 - D))
        part2 = np.cos(2 ** i * np.pi * x)
        w += part1 * part2
    return w


def add_asym_noise(RR, sd1, sd2):
    differences = np.diff(RR)
    RR_noise = []
    for idx in range(len(RR) - 1):
        if differences[idx] > 0:
            RR_noise.append(RR[idx + 1] + np.random.normal(0, sd1))
        elif differences[idx] < 0:
            RR_noise.append(RR[idx + 1] + np.random.normal(0, sd2))
    return np.array(RR_noise)


def asymmetric_weierstrass_jp(length=10000, D=None, sd1=0.5, sd2=2.0):
    if D is None:
        D = np.random.uniform(1.2, 1.8)
    return add_asym_noise(weierstrass(length, D), sd1, sd2)

def generate_normal_signal(mean, std_dev, num_samples):
    signal = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    return signal


def generate_chisquare_signal(df, num_samples):
    signal = np.random.chisquare(df=df, size=num_samples)
    return signal


if __name__ == '__main__':
    normal = generate_normal_signal(0,sigma*scale,n)
    chi_square = generate_chisquare_signal(df,n)
    ar = autoregressive_function(n,phi,sigma,scale)
    ar_asymmetric = asymmetric_autoregressive_jp(n,phi,sigma, scale)
    aw_classic = asymmetric_weierstrass_sawtooth(a,b,n)

    x = np.linspace(-2, 2, 1000)
    aw_jp = asymmetric_weierstrass_jp(length=500)
    from src.core.runs.runs_entropy import Signal, Runs

    signal = Signal(chi_square, np.zeros_like(chi_square))
    runs = Runs(signal)
    dec_entropy = runs.HDR
    acc_entropy = runs.HAR
    neutral_entropy = runs.HNO
    print(dec_entropy,acc_entropy,neutral_entropy)

    plt.figure(figsize=(12, 6))
    plt.plot(ar, label='Generated AR(1) Series')
    plt.plot(ar_asymmetric, label='Generated asymmetric AR(1) Series')
    plt.plot(aw_classic,  label='Generated asymmetric weierstrass (classic) Series')
    plt.plot(aw_jp,  label='Generated asymmetric weierstrass (JP) Series')

    plt.plot(normal, label="Generated normal distribution Series")
    plt.plot(chi_square, label="Generated chi square distribution Series")

    plt.title('Generated AR(1) Series')
    plt.legend()
    plt.show()