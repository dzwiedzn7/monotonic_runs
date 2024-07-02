import glob

import numpy as np

from runs_entropy import RRLoaderNoAnnotations, Signal, Runs

if __name__ == '__main__':
    sr_tachograms = glob.glob("ltafdb_rr_60s/ltafdb_rr_60s/sr/*")
    af_tachograms = glob.glob("ltafdb_rr_60s/ltafdb_rr_60s/af/*")

    filename = "ltafdb_rr_60s/ltafdb_rr_60s/af/00_af_10_0.csv"
    raw_rr = np.loadtxt(filename, skiprows=0, delimiter="\t")
    #print(np.average(raw_rr))
    #print(np.std(raw_rr))

    rr_diff = abs(raw_rr[1:] - raw_rr[:-1])
    #print(rr_diff)
    nnx = rr_diff[np.where(rr_diff>30.)]
    pnnx = float(len(nnx))/float(len(rr_diff)) *100
    #print(pnnx)

    nnxpercent = rr_diff[np.where(rr_diff> raw_rr[:-1] * 3.25/100)]
    pnnxpercent = float(len(nnxpercent))/float(len(rr_diff)) *100
    #print(pnnxpercent)
    n = raw_rr.size
    rr = np.where((raw_rr < 3000) & (raw_rr > 240), raw_rr, 0)

    #print(rr)
    annotations = np.where((raw_rr > 3000) | (raw_rr < 240), raw_rr, 0)
    annotations[annotations>0] = 1
    #print(annotations)
    non_zero_idx = np.nonzero(rr)
    rr = rr[non_zero_idx]

    annotations_sr = np.zeros(len(rr))
    #annotations_sr = np.array([1,0,0,0,1,0,1,0,1,0,1,0,0,0,1])
    signal_sr = Signal(raw_rr, annotations=annotations)

    runs_sr = Runs(signal_sr)
    #print(runs_sr.signal)
    #print(len(runs_sr.signal.rr))
    decc_runs_sr = runs_sr.count_for_all(">")
    acc_runs_sr = runs_sr.count_for_all("<")
    neutral_runs = runs_sr.count_for_all("==")
    print(decc_runs_sr)
    print(acc_runs_sr)
    #print(runs_sr.jp_entropy("<", True))
    print(neutral_runs)

    dec_entropy = runs_sr.shannon_entropy(">","rr")
    acc_entropy = runs_sr.shannon_entropy("<","rr")
    net_entropy = runs_sr.shannon_entropy("==","rr")
    #print(dec_entropy)
    #print(acc_entropy)
    #print(net_entropy)
    print("JP")
    print(runs_sr.jp_entropy("<")) #HDR
    print(runs_sr.jp_entropy(">")) #HAR
    print(runs_sr.jp_entropy("==", bidirection=False))

    #bi = runs_sr.bidirectional_count()
    #print(bi)
    #bi_shannon_sr = runs_sr.shannon_entropy_bidirectional()
    #print(bi_shannon_sr)



    """
    for idx in range(2):
        #test_path = sr_tachograms[0]

        rr_sr = RRLoaderNoAnnotations().load(sr_tachograms[idx])
        #print(rr)
        annotations_sr = np.zeros(len(rr_sr))
        signal_sr = Signal(rr_sr, annotations=annotations_sr)
        runs_sr = Runs(signal_sr)
        decc_runs_sr = runs_sr.count_for_all(">")
        acc_runs_sr = runs_sr.count_for_all("<")
        #neutral_runs = runs.count_for_all("==")
        shannon_acc_sr = runs_sr.shannon_entropy("<") # entropie z rr czy normalną?
        shannon_decc_sr = runs_sr.shannon_entropy(">")
        #neutral_runs = noised_runs.count_for_all("==")
        #print(runs.signal.rr)


        rr_af = RRLoaderNoAnnotations().load(af_tachograms[idx])
        #print(rr)
        annotations_af = np.zeros(len(rr_af))
        signal_af = Signal(rr_af, annotations=annotations_af)
        runs_af = Runs(signal_af)
        decc_runs_af = runs_af.count_for_all(">")
        acc_runs_af = runs_af.count_for_all("<")
        #neutral_runs = runs.count_for_all("==")
        shannon_acc_af = runs_af.shannon_entropy("<", 'rr')
        shannon_decc_af = runs_af.shannon_entropy(">", 'rr')
        #neutral_runs = noised_runs.count_for_all("==")
        #print(runs.signal.rr)

        bi_shannon_af = runs_af.shannon_entropy_bidirectional()
        print(bi_shannon_af)

        bi_shannon_sr = runs_sr.shannon_entropy_bidirectional()
        print(bi_shannon_sr)


        print(decc_runs_sr, decc_runs_af)
        print(acc_runs_sr, acc_runs_af)
        print(shannon_acc_sr, shannon_acc_af,"acc entropy diff: ",abs(shannon_acc_sr-shannon_acc_af))
        print(shannon_decc_sr, shannon_decc_af,"dec entropy diff: ",abs(shannon_decc_sr-shannon_decc_af))
        #print(bi_shannon_sr, bi_shannon_af,"bi entropy diff: ", abs(bi_shannon_sr-bi_shannon_af))
    """