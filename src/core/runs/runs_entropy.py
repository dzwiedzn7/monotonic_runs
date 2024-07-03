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
    """
    def __add__(self, other):
            if not isinstance(other, Counter):
                return NotImplemented
            result = Counter()
            for elem, count in sorted(self.items()):
                newcount = count + other[elem]
                result[elem] = newcount
            for elem, count in sorted(other.items()):
                if elem not in self:
                    result[elem] = count
            return result
    """

    #def keys() -> return sorted keys

    #def counts() -> return counts


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
        rr = data[:, 1]
        noise = np.random.normal(0, 0.001, len(rr))
        noised_rr = rr + noise
        annotations = data[:, 2]
        return noised_rr, annotations


class RRLoaderNoAnnotations(AbstractDataLoader):
    def load(self, filename):
        rr = np.loadtxt(filename, skiprows=1, delimiter="\t")
        return rr


class Runs:
    def __init__(self, signal: Signal):
        self.signal = signal
        #self.n_rr = self.total_number_of_rr()
        #self.runs_acc = self.count_for_all("<")
        #self.runs_dec = self.count_for_all(">")
        #self.runs_neutral = self.count_for_all("==")
        #self.runs_bidirectional = self.runs_dec + self.runs_acc + self.runs_neutral

    def split_signal_into_segments(self):
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

    def split_runs(self, operator):
        splited_runs = []
        signal_segments = self.split_signal_into_segments()
        for segment in signal_segments:
            if len(segment) < 2:
                continue
            mask = np.full((len(segment)), False)
            split_points = np.where(self.diff_conditions(segment, operator))[0] + 1
            mask[split_points] = True
            run = self.splitByBool(segment, mask)
            splited_runs.append(run)
        return splited_runs

    def count_for_all(self, operator):
        collect_list = []
        runs = self.split_runs(operator)
        for segment in runs:
            if len(segment) > 0:
                run_count = [len(run) for run in segment]
                collect_list += run_count
        counter = OrderedCounter(collect_list)
        return counter

    # depricated
    def bidirectional_count(self):
        acc = self.count_for_all(">")
        dec = self.count_for_all("<")
        neutral = self.count_for_all("==")
        return acc + dec + neutral

    def sum_of_all_runs(self):
        neutral = self.count_for_all("==").values()
        acc = self.count_for_all("<").values()
        dec = self.count_for_all(">").values()
        return sum(neutral) + sum(acc) + sum(dec)

    def total_number_of_rr(self):
        return sum([len(segment) for segment in self.split_signal_into_segments()])

    def probability(self, operator):
        counter = self.count_for_all(operator)
        run_counts = list(dict(sorted(counter.items())).values())
        total = self.sum_of_all_runs()
        probability = [run_count/total for run_count in run_counts]
        return probability, sum(probability)

    def rr_probability(self, operator):
        counter = self.count_for_all(operator)
        run_counts = list(dict(sorted(counter.items())).values())
        run_lenght = sorted(counter.keys())
        total = self.total_number_of_rr()
        entropy = [len*run_count/total for run_count, len in zip(run_counts, run_lenght)]
        return entropy, sum(entropy)

    def shannon_entropy(self, operator, entropy_type=None):
        if entropy_type == "rr":
            entropy = self.rr_probability(operator)
        else:
            entropy = self.probability(operator)
        shannon = -sum([ent*np.log(ent) for ent in entropy[0]])
        return shannon

    def jp_entropy(self, operator, bidirection=False):
        # - i * counts[i]/n * log(i * counts[i]/n)

        entropy = 0

        counter = self.count_for_all(operator)
        #print(counter)
        run_counts = list(dict(sorted(counter.items())).values())
        run_lenght = sorted(counter.keys())
        total = self.total_number_of_rr()
        n_keys = sorted(self.bidirectional_count().keys())
        n_keys = list(range(n_keys[0],n_keys[-1]+1))
        #n_vals = sorted(self.bidirectional_count().values())[::-1]
        n_vals = [self.bidirectional_count()[key] for key in n_keys]
        #print(n_vals)
        #print(n_keys)
        #print(n_vals[::-1])
        #print(n_keys)
        n = sum([x * (i + 1) for i, x in enumerate(list(dict(sorted(self.count_for_all(">").items())).values()))] +
                [x * (i + 1) for i, x in enumerate(list(dict(sorted(self.count_for_all("<").items())).values()))] +
                [x * (i + 1) for i, x in enumerate(list(dict(sorted(self.count_for_all("==").items())).values()))])
        #print(n)
        n =sum([key*val for key,val in zip(n_keys,n_vals)])
        #print(n)
        for run_count,len in zip(run_counts,run_lenght):
            #print(run_count,len)
            entropy += -len*run_count/total * np.log(len*run_count/total)

        if bidirection:
            jp_entropy = self.individual_entropy(self.bidirectional_count(), n)
        else:
            jp_entropy = self.individual_entropy(counter, n)
        return jp_entropy


    def individual_entropy(self, counter, n):
        #print(counter)
        help = list(sorted(counter.keys()))
        #print(help)
        if len(help) == 0:
            return 0
        help_range = list(range(help[0],help[-1]+1))
        #print(help_range)
        #print(list(range(help[0],help[-1]+1)))
        import math
        full = 0
        partial = 0
        for i in help_range:
            if i in counter.keys():
                #print(counter[i],i,n)
                partial = - counter[i] * i / n * math.log(counter[i] * i / n)
            full += partial
        return full




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
    rr, annotations = RRLoader().load("F:\phd\monotonic_runs\src\core\\tests\\resources\\rr\\1.txt",0,1)
    #noised_rr, noised_annotations = NoisedRRLoader().load("0001.rea")

    #annotations =  np.zeros(len(rr))
    signal = Signal(rr, annotations)
    #noised_signal = Signal(noised_rr, noised_annotations)
    runs = Runs(signal)

    #noised_runs = Runs(noised_signal)
    decc_runs = runs.count_for_all(">")
    acc_runs = runs.count_for_all("<")
    neutral_runs = runs.count_for_all("==")
    dec_entropy = runs.jp_entropy("<")
    acc_entropy = runs.jp_entropy(">")
    neutral_entropy = runs.jp_entropy("==")
    #neutral_runs = noised_runs.count_for_all("==")
    print(decc_runs)
    print(acc_runs)
    print(neutral_runs)
    print("HDR: ",dec_entropy)
    print("HAR: :",acc_entropy)
    print("HNR: ",neutral_entropy)
    #print(runs.bidirectional_count())
    #plt.plot(rr)
    #annotations_plot = [(r*a)/a+0.1 for a,r in  zip(annotations,rr)]
    #plt.scatter(list(range(len(annotations))),annotations_plot)
    #plt.show()
    #print(runs.signal.rr)
    #print(noised_runs.signal.rr)
    #for i in range(len(runs.signal.rr)):
    #    print(runs.signal.rr[i],noised_runs.signal.rr[i],runs.signal.rr[i] - noised_runs.signal.rr[i])
#21	14	5	1	1
#22	17	4	1	1	0	0	1
