import numpy as np
import csv

def process():
    # Parameters
    D = 1.5
    N = 20000  # length of the time series
    t = np.linspace(0, 2*np.pi, N, endpoint=False)  # sample times from 0 to 2π

    # Number of terms in the truncated sum
    num_terms = 20

    alpha = 2 - D  # alpha = 0.5
    W = np.zeros_like(t)

    # Generate W(t) as a sum of cosines
    for n in range(num_terms):
        scale = 2**(-(2 - D)*n)
        W += scale * np.cos((2**n) * t)

    # Now W is our deterministic time series
    # We want to add noise depending on whether W(t_i) > W(t_{i-1}) or not.

    k = 2.0  # choose some value for k
    #k = np.random.randint(1, 11)  # m in {1, 2, ..., 10}

    # Create a new series g, which is W plus noise
    g = W.copy()

    # Add noise increments:
    # If W(i) > W(i-1), add N(0, 1+1/k)
    # If W(i) < W(i-1), add N(0, 1-1/k)
    # For i = 0, there's no previous point, so we skip or assume it's always an increase/decrease.

    for i in range(1, N):
        if W[i] > W[i-1]:
            # Increase
            g[i] += np.random.normal(0, np.sqrt(1 + 1/k))
        else:
            # Decrease
            g[i] += np.random.normal(0, np.sqrt(1 - 1/k))

    # Compute the empirical variance of g
    empirical_mean_g = np.mean(g)
    empirical_var_g = np.mean((g - empirical_mean_g)**2)

    #print("Empirical variance of g:", empirical_var_g)
    return g

# Save g and a column of zeros to a CSV file
#output_csv = 'weierstrass_asym_noise.csv'
#zeros_column = np.zeros_like(g)

#with open(output_csv, mode='w', newline='') as file:
#    writer = csv.writer(file)
#    writer.writerow(['g_signal', 'zeros'])  # Header
#    writer.writerows(zip(g, zeros_column))

#print(f"Signal saved to {output_csv}")

import scipy.stats as stats

def calculate_sd1a_sd1d(rr_intervals):
    n = len(rr_intervals)
    y = rr_intervals[1:]
    x = rr_intervals[:-1]
    xy = ((x-np.mean(x)-y+np.mean(y))/np.sqrt(2))
    dec = np.array([val for val in xy if val < 0])
    acc = np.array([val for val in xy if val > 0])
    sd1a = np.sqrt(sum(np.array(acc)**2)/n)
    sd1d = np.sqrt(sum(np.array(dec)**2)/n)

    return sd1a, sd1d

sda1a_bigger_list = []
sd1a_list,sd1d_list = [],[]
har,hdr,hno = [],[],[]

for _ in range(100):
    g = process()
    #for sub in np.array_split(g,100):
    #    annotations = np.zeros(len(sub))
    sd1a,sd1d = calculate_sd1a_sd1d(g)
    sd1a_list.append(sd1a)
    sd1d_list.append(sd1d)
    annotations = np.zeros(len(g))
    signal = Signal(g, annotations=annotations)
    sd1a,sd1d = calculate_sd1a_sd1d(g)
    runs = Runs(signal)
    har.append(runs.HAR)
    hdr.append(runs.HDR)
    hno.append(runs.HNO)

sd1a_bigger = np.count_nonzero(np.array(sd1a_list) > np.array(sd1d_list))
    #sda1a_bigger_list.append(sd1a_bigger)
print(sd1d_list)
print(sd1a_list)
#median = np.median(np.array(sda1a_bigger_list))
#mean = np.mean(np.array(sda1a_bigger_list))
#print("median", median)
#print("mean", mean)
#p_value = stats.binom_test(median, n=100, p=0.5)
#print("median p value",p_value)
#p_value = stats.binom_test(mean, n=100, p=0.5)
#print("mean p value",p_value)

p_value = stats.binom_test(sd1a_bigger,n=100, p=0.5)
print("p_value", p_value)
print("sd1a bigger:", sd1a_bigger)