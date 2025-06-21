import numpy as np

# Parameters
D = 1.5
N = 1000  # length of the time series
t = np.linspace(0, 2*np.pi, N, endpoint=False)  # sample times from 0 to 2π

# Number of terms in the truncated sum
num_terms = 20

# Precompute the factor for each n to save time
# factor_n = 1 / (2^{(2-D)*n}) = 2^{-(2-D)*n}
alpha = 2 - D  # alpha = 0.5
# We'll build W(t) as a sum of cosines
W = np.zeros_like(t)

for n in range(num_terms):
    # frequency: 2^n
    # amplitude scaling: 2^{-(2-D)*n}
    scale = 2**(-(2 - D)*n)
    W += scale * np.cos((2**n) * t)

# Now, W is our generated time series
# Compute the empirical variance
empirical_mean = np.mean(W)
empirical_var = np.mean((W - empirical_mean)**2)

print("Empirical variance:", empirical_var)

# Theoretical variance:
# Var(W) = 1/(2(1 - 2^{-2(2-D)}))
theoretical_var = 1/(2*(1 - 2**(-2*(2-D))))

print("Theoretical variance:", theoretical_var)
