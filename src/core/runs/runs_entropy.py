import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import Counter, OrderedDict
from typing import Optional


class OrderedCounter(Counter, OrderedDict):

    def __add__(self, other):
        """Add counts from two ordered counters.

        >>> Counter('abbb') + Counter('bcc')
        Counter({'b': 4, 'c': 2, 'a': 1})

        """
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()

        for elem, count in sorted(self.items()):
            newcount = count + other[elem]
            if newcount > 0:
                result[elem] = newcount
        for elem, count in sorted(other.items()):
            if elem not in self and count > 0:
                result[elem] = count
        return result


@dataclass
class Signal:
    rr: np.ndarray
    annotations: Optional[np.ndarray] = None

    """
    @property
    def filter_rr(self):
        filtered_rr = np.where((self.rr < 3000) & (self.rr > 240), self.rr, 0)
        non_zero_idx = np.nonzero(filtered_rr)
        return self.rr[non_zero_idx]

    @property
    def percent_of_rejected_rr(self):
        n = len(self.rr)
        passed = len(self.filter_rr)
        return 1-(passed/n)

    @property
    def sdnn(self):
        return np.std(self.filter_rr)

    @property
    def average_rr(self):
        return np.average(self.filter_rr)
    """

class AbstractDataLoader(ABC):
    @abstractmethod
    def load(self, filename, rr_col, an_col):
        """implements how to load data"""


class RRLoader(AbstractDataLoader):
    def load(self, filename, rr_col, an_col):
        data = np.loadtxt(filename, skiprows=1, delimiter="\t")
        rr = data[:, rr_col]
        annotations = data[:, an_col]
        return rr, annotations


class NoisedRRLoader(AbstractDataLoader):
    def load(self, filename, rr_col, an_col):
        data = np.loadtxt(filename, skiprows=1, delimiter="\t")
        rr = data[:, rr_col]
        noise = np.random.normal(0, 0.001, len(rr))
        noised_rr = rr + noise
        annotations = data[:, an_col]
        return noised_rr, annotations


class RRLoaderNoAnnotations(AbstractDataLoader):
    def load(self, filename, rr_col, an_col):
        rr = np.loadtxt(filename, skiprows=1, delimiter="\t")
        return rr


class RRLoaderShuffled(AbstractDataLoader):
    def load(self, filename, rr_col, an_col):
        data = np.loadtxt(filename, skiprows=1, delimiter="\t")
        np.random.shuffle(data)
        rr = data[:, rr_col]
        annotations = data[:, an_col]
        return rr, annotations

class Runs:
    def __init__(self, signal: Signal):
        self.signal = signal
        self.segments = self.split_signal_into_segments()
        self.operators = ["<", ">", "=="]
        self.runs_cache = {}
        self.counters_cache = {}
        self.create_runs_counter()
        self.HDR = self.asymmetrical_entropy(self.counters_cache["<"])
        self.HAR = self.asymmetrical_entropy(self.counters_cache[">"])
        self.HNO = self.asymmetrical_entropy(self.counters_cache["=="])

    def split_signal_into_segments(self):
        #if self.signal.annotations is None:
        #    return np.array([self.signal.rr])
        bad_indices = np.where(self.signal.annotations != 0.0)[0]

        start = 1
        signal_segments = []
        for idx in bad_indices:
            end = idx
            if start < end:
                signal_segments.append(self.signal.rr[start-1:end])
            start = idx +1
        if self.signal.annotations[len(self.signal.rr) - 1] == 0:
            signal_segments.append(self.signal.rr[start:len(self.signal.rr)])
        return signal_segments

    def split_runs(self):
        for operator in self.operators:
            splited_runs = []
            for segment in self.segments:
                if len(segment) < 2:
                    continue
                mask = np.full((len(segment)), False)
                split_points = np.where(self.diff_conditions(segment, operator))[0] + 1
                mask[split_points] = True
                run = self.splitByBool(segment, mask)
                splited_runs.append(run)
            self.runs_cache[operator] = splited_runs
        return self.runs_cache

    def create_runs_counter(self):
        self.split_runs()  # Ensure runs_cache is populated
        for operator, run_type in self.runs_cache.items():
            collect_list = []
            for segment in run_type:
                if len(segment) > 0:
                    run_count = [len(run) for run in segment]
                    collect_list += run_count
            counter = Counter(collect_list)
            self.counters_cache[operator] = counter
        return self.counters_cache

    # depricated
    @property
    def total_run_counter(self):
        acc = self.counters_cache[">"]
        dec = self.counters_cache["<"]
        neutral = self.counters_cache["=="]
        return acc + dec + neutral

    @property
    def sum_of_all_runs(self):
        acc = self.counters_cache[">"].values()
        dec = self.counters_cache["<"].values()
        neutral = self.counters_cache["=="].values()
        return sum(neutral) + sum(acc) + sum(dec)

    @property
    def total_number_of_rr(self):
        return sum([len(segment) for segment in self.segments])

    def probability(self, runs_counter):
        run_counts = list(dict(sorted(runs_counter.items())).values())
        total = self.sum_of_all_runs()
        probability = [run_count/total for run_count in run_counts]
        return probability, sum(probability)

    #depricated
    def rr_probability(self, runs_counter):
        run_counts = list(dict(sorted(runs_counter.items())).values())
        run_lenght = sorted(runs_counter.keys())
        total = self.total_number_of_rr
        entropy = [len*run_count/total for run_count, len in zip(run_counts, run_lenght)]
        return entropy, sum(entropy)

    def shannon_entropy(self, runs_counter, entropy_type=None):
        if entropy_type == "rr":
            entropy = self.rr_probability(runs_counter)
        else:
            entropy = self.probability(runs_counter)
        shannon = -sum([ent*np.log(ent) for ent in entropy[0]])
        return shannon

    def asymmetrical_entropy(self, runs_counter):
        # Compute n based on the total_run_counter
        n_keys = sorted(self.total_run_counter.keys())
        n_range = list(range(n_keys[0], n_keys[-1] + 1))
        n_vals = [self.total_run_counter.get(key, 0) for key in n_range]
        n = sum(key * val for key, val in zip(n_range, n_vals))

        sorted_keys = sorted(runs_counter.keys())
        if not sorted_keys:
            return 0

        entropy = 0
        for i in range(sorted_keys[0], sorted_keys[-1] + 1):
            if i in runs_counter:
                count = runs_counter[i]
                partial_entropy = -count * i / n * np.log(count * i / n)
                entropy += partial_entropy

        return entropy

    @staticmethod
    def diff_conditions(array, operator):
        if operator == "<":
            return np.diff(array) < 0
        elif operator == ">":
            return np.diff(array) > 0
        elif operator == "==":
            return np.diff(array) == 0

    @staticmethod
    def splitByBool(a, m):
        if m[0]:
            return np.split(a, np.nonzero(np.diff(m))[0] + 1)[::2]
        else:
            return np.split(a, np.nonzero(np.diff(m))[0] + 1)[1::2]



if __name__ == "__main__":
    rr, annotations = RRLoader().load(u"src/core/tests/resources/rr/1.txt",0,1)
    #noised_rr, noised_annotations = NoisedRRLoader().load("0001.rea")

    #annotations =  np.zeros(len(rr))
    signal = Signal(rr, annotations)
    #noised_signal = Signal(noised_rr, noised_annotations)
    runs = Runs(signal)

    #noised_runs = Runs(noised_signal)
    #decc_runs = runs.count_for_all(">")
    #acc_runs = runs.count_for_all("<")
    #neutral_runs = runs.count_for_all("==")
    #dec_entropy = runs.asymmetrical_entropy("<")
    #acc_entropy = runs.asymmetrical_entropy(">")
    #neutral_entropy = runs.asymmetrical_entropy("==")
    #neutral_runs = noised_runs.count_for_all("==")
    print(runs.counters_cache["<"])
    print(runs.counters_cache[">"])
    print(runs.counters_cache["=="])
    print("HDR: ",runs.HDR)
    print("HAR: :",runs.HAR)
    print("HNR: ",runs.HNO)
