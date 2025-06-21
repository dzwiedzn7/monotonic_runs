import numpy as np
import csv
# Parameters
def process():
    D = 1.5
    N = 10000  # length of the time series
    t = np.linspace(0, 2*np.pi, N, endpoint=False)  # sample times from 0 to 2π

    # Number of terms in the truncated sum
    num_terms = 20
    alpha = 2 - D  # alpha = 0.5

    # Baseline W(t) as given
    W = np.zeros_like(t)
    for n in range(num_terms):
        scale = 2**(-(2 - D)*n)
        W += scale * np.cos((2**n) * t)

    # Select m at random from the range [1, 10]
    m = np.random.randint(1, 11)  # m in {1, 2, ..., 10}

    # Create a new process g(t_i)
    g = np.zeros_like(W)
    g[0] = W[0]  # initial value

    # Iterate through the series and add the appropriate noise
    for i in range(1, N):
        if W[i] > W[i-1]:
            # Add N(0,1)
            increment = np.random.normal(0, 1)
        else:
            # Add sum of m normals, each N(0, 1/m)
            increments = np.random.normal(0, np.sqrt(1/m), size=m)
            increment = np.sum(increments)
        g[i] = W[i] + increment

    # Compute empirical variance of g
    empirical_mean_g = np.mean(g)
    empirical_var_g = np.mean((g - empirical_mean_g)**2)
    #print("Empirical variance of g:", empirical_var_g)
    #print("Chosen m:", m)
    return g
# Save g and a column of zeros to a CSV file
#output_csv = 'weierstrass_modified_runs.csv'
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
for _ in range(100):
    g = process()

    #for sub in np.array_split(g,100):
    #    annotations = np.zeros(len(sub))
    sd1a,sd1d = calculate_sd1a_sd1d(g)
    sd1a_list.append(sd1a)
    sd1d_list.append(sd1d)
print(sd1a_list)
print(sd1d_list)
sd1a_bigger = np.count_nonzero(np.array(sd1a_list) > np.array(sd1d_list))
    #sda1a_bigger_list.append(sd1a_bigger)

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