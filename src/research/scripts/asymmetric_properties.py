import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from src.core import weierstrass as weier
from src.core.runs.runs_entropy import RRLoader, RRLoaderShuffled, Signal, Runs
import seaborn as sns
import statsmodels.api as sm

def resource_files():
    resource_dir = "../data/rr"
    file_paths = []

    for root, dirs, files in os.walk(resource_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths


def get_entropty_from_shufled_rr(resources):
    har, hdr, hno = [], [], []
    for name in resources:
        rr, annotations = RRLoaderShuffled().load(name,0,1)
        signal = Signal(rr, annotations)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])

def get_entropty_from_rr(resources):
    har, hdr, hno = [], [], []
    for name in resources:
        rr, annotations = RRLoader().load(name,0,1)
        signal = Signal(rr, annotations)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])

def get_entropy_from_weierstrass_jp(num_samples, len_sample):
    a = 0.5
    b = 3
    n_terms = 50  # Number of terms in the serie for Weierstrass

    har, hdr, hno = [], [], []
    np.random.seed(seed=int(time.time()))
    for idx in range(num_samples):
        x = np.linspace(-2, 2, len_sample) + np.random.normal(0, 0.1, len_sample)
        signal = weier.asymmetric_weierstrass_jp(x, a, b, n_terms)
        anot = np.zeros(len(signal))
        signal = Signal(signal,annotations=anot)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])

def get_entropy_from_weierstrass_sawtooth(num_samples, len_sample):
    a = 0.5
    b = 3

    har, hdr, hno = [], [], []
    np.random.seed(seed=int(time.time()))
    for idx in range(num_samples):
        signal = weier.asymmetric_weierstrass_sawtooth(a, b, len_sample)
        anot = np.zeros(len(signal))
        signal = Signal(signal, annotations=anot)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])

def get_entropy_from_AR(num_samples, len_sample):
    n = len_sample  # Number of data points
    phi = 0.1  # AR parameter
    sigma = 0.1  # Standard deviation of noise
    scale = 3

    har, hdr, hno = [], [], []
    np.random.seed(seed=int(time.time()))
    for idx in range(num_samples):
        signal = weier.autoregressive_function(n, phi, sigma, scale)
        anot = np.zeros(len(signal))
        signal = Signal(signal, annotations=anot)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])

def get_entropy_from_AR_asymmetric(num_samples, len_sample):
    n = len_sample  # Number of data points
    phi = 0.1  # AR parameter
    sigma = 0.1  # Standard deviation of noise
    scale = 3

    har, hdr, hno = [], [], []
    np.random.seed(seed=int(time.time()))
    for idx in range(num_samples):
        aw_sawtooth = weier.autoregressive_function(n, phi, sigma, scale)
        anot = np.zeros(len(aw_sawtooth))
        signal = Signal(aw_sawtooth, annotations=anot)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])

def get_entropy_from_normal(num_samples, len_sample):
    mean = 0
    sigma = 0.1  # Standard deviation of noise
    scale = 3
    std_dev = sigma * scale

    har, hdr, hno = [], [], []
    #np.random.seed(seed=int(time.time()))
    for idx in range(num_samples):
        signal = np.random.normal(loc=mean, scale=std_dev, size=len_sample)
        anot = np.zeros(len(signal))
        signal = Signal(signal, annotations=anot)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])

def get_entropy_from_chi_square(num_samples, len_sample):
    df = 2  # The degrees of freedom of the chi-square distribution.

    har, hdr, hno = [], [], []
    # np.random.seed(seed=int(time.time()))
    for idx in range(num_samples):
        signal = np.random.chisquare(df=df, size=len_sample)
        anot = np.zeros(len(signal))
        signal = Signal(signal, annotations=anot)
        runs = Runs(signal)
        har.append(runs.HAR)
        hdr.append(runs.HDR)
        hno.append(runs.HNO)
    return pd.DataFrame(np.array([har, hdr, hno]).T, columns=['HAR', 'HDR', 'HNO'])


def statistical_tests(group1, group2):
    stat1, p1 = stats.shapiro(group1)
    stat2, p2 = stats.shapiro(group2)
    if 0.05 <= p1 and 0.05 <= p2:
        print("Performing T-test for independent groups")
        return stats.ttest_ind(group1, group2)
    else:
        print("Performing Mann-Whitney test")
        return stats.mannwhitneyu(group1, group2)

def bootstrap_p_values(group1, group2, n_iterations=2):
    p_values = []
    for _ in range(n_iterations):
        shuffled_group2 = get_entropty_from_shufled_rr(resources)
        _, p_value = statistical_tests(group1, shuffled_group2[group2.name])
        p_values.append(p_value)
    return p_values



if __name__ == "__main__":
    resources = resource_files()
    columns = ["HAR", "HDR", "HNO"]

    non_shuffled_entropy_df = get_entropty_from_rr(resources)
    shuffled_entropy_df = get_entropty_from_shufled_rr(resources)

    # Histograms, Q-Q plots, and Boxplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 25))

    for i, column in enumerate(columns):
        # Histogram
        sns.histplot(non_shuffled_entropy_df[column], label="Non-Shuffled", ax=axes[0, i], kde=True)
        sns.histplot(shuffled_entropy_df[column], label="Shuffled", ax=axes[0, i], kde=True)
        axes[0, i].set_xlabel(column)
        axes[0, i].set_title(f"Histogram of {column}")
        axes[0, i].legend()

        # Q-Q plot
        sm.qqplot(non_shuffled_entropy_df[column], line='45', ax=axes[1, i], label="Non-Shuffled")
        sm.qqplot(shuffled_entropy_df[column], line='45', ax=axes[1, i], label="Shuffled")
        axes[1, i].set_title(f"Q-Q Plot of {column}")
        axes[1, i].legend()

        # Boxplot
        data_to_plot = [non_shuffled_entropy_df[column], shuffled_entropy_df[column]]
        axes[2, i].boxplot(data_to_plot, labels=['Non-Shuffled', 'Shuffled'])
        axes[2, i].set_title(f"Boxplot of {column}")
        axes[2, i].set_ylabel(column)

    plt.suptitle("Histogram, Q-Q plots, and Boxplots of Entropy for shuffled and non-shuffled RR intervals")
    plt.tight_layout()
    plt.show()

    # Correlation analysis
    print("Correlation analysis for non-shuffled data:")
    print(non_shuffled_entropy_df.corr())
    print("\nCorrelation analysis for shuffled data:")
    print(shuffled_entropy_df.corr())

    # Variation analysis
    print("\nVariation analysis for non-shuffled data:")
    print(non_shuffled_entropy_df.var())
    print("\nVariation analysis for shuffled data:")
    print(shuffled_entropy_df.var())

    # Statistical tests and bootstrap
    for column in columns:
        print(f"\nStatistical tests for {column}:")
        stat_result = statistical_tests(non_shuffled_entropy_df[column], shuffled_entropy_df[column])
        print(stat_result)

        print(f"\nBootstrap p-values for {column}:")
        p_values = bootstrap_p_values(non_shuffled_entropy_df[column], shuffled_entropy_df[column])
        print(f"Mean p-value: {np.mean(p_values)}")
        print(f"95% CI: ({np.percentile(p_values, 2.5)}, {np.percentile(p_values, 97.5)})")

    # Additional statistical analysis: Effect size (Cohen's d)
    for column in columns:
        effect_size = (non_shuffled_entropy_df[column].mean() - shuffled_entropy_df[column].mean()) / \
                      np.sqrt((non_shuffled_entropy_df[column].var() + shuffled_entropy_df[column].var()) / 2)
        print(f"\nCohen's d effect size for {column}: {effect_size}")

    # Kolmogorov-Smirnov test for distribution comparison
    for column in columns:
        ks_stat, ks_p = stats.ks_2samp(non_shuffled_entropy_df[column], shuffled_entropy_df[column])
        print(f"\nKolmogorov-Smirnov test for {column}:")
        print(f"Statistic: {ks_stat}, p-value: {ks_p}")

    #print(shuffled_entropy_df)

    #weierstrass_jp_df = get_entropy_from_weierstrass_jp(len(resources), 30000)
    #print(weierstrass_jp_df)

    #weierstrass_sawtooth_df = get_entropy_from_weierstrass_sawtooth(len(resources), 30000)
    #print(weierstrass_sawtooth_df)

    #ar_classic_df = get_entropy_from_AR(len(resources),30000)
    #print(ar_classic_df)

    #ar_asymmetric_df = get_entropy_from_AR_asymmetric(len(resources), 30000)
    #print(ar_asymmetric_df)


    #normal_df = get_entropy_from_normal(len(resources), 30000)
    #print(normal_df)

    #chi_square_df = get_entropy_from_chi_square(len(resources), 30000)
    #print(chi_square_df)

    #print(stats.pearsonr(chi_square_df["HAR"],chi_square_df["HDR"])) #PearsonRResult(statistic=-0.15621069847010557, pvalue=0.40137260554367576)
    #shuffled_df = chi_square_df.sample(frac=1).reset_index(drop=True)
    #print(stats.pearsonr(shuffled_df["HAR"], shuffled_df["HDR"]))