# monotonic runs (shuffle)
# monotonic runs (non shuffle)
# asymmetric weierstrass (jp) done
# asymmetric weierstrass (classic) done
#  autoregression AR done
# normal distribution
# chi square distribution

import numpy as np
import matplotlib.pyplot as plt


def weierstrass_function(x, a, b, n_terms):
    """
    Generate a signal using the Weierstrass function.

    Parameters:
    x (np.ndarray): Input array of x values.
    a (float): Parameter a, where 0 < a < 1.
    b (int): Parameter b, a positive odd integer.
    n_terms (int): Number of terms in the series.

    Returns:
    np.ndarray: The Weierstrass function evaluated at each x.
    """
    W = np.zeros_like(x)
    for n in range(n_terms):
        W += a ** n * np.cos(b ** n * np.pi * x)
    return W


# Example usage
a = 0.5
b = 7
n_terms = 50
x = np.linspace(-2, 2, 1000)
signal = weierstrass_function(x, a, b, n_terms)

# Plot the signal
plt.plot(x, signal)
plt.title('Weierstrass Function')
plt.xlabel('x')
plt.ylabel('W(x)')
plt.grid(True)
plt.show()