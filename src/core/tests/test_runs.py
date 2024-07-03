import unittest
import os
import pytest
import pandas as pd
from src.core.runs.runs_entropy import RRLoader, Signal, Runs


class TestResources:
    @pytest.mark.usefixtures("resource_files")
    def test_resource_files(self, resource_files):
        assert len(resource_files) > 0  # Ensure there are files in the directory
        for file_path in resource_files:
            assert os.path.exists(file_path)


class TestRuns:
    @pytest.mark.usefixtures("ground_truth_df")
    @pytest.mark.usefixtures("resource_files")
    def test_entropy(self, resource_files, ground_truth_df):
        for idx, file_path in enumerate(resource_files):
            print(idx,file_path)
            truth = ground_truth_df.iloc[idx]
            rr, annotations = RRLoader().load(file_path, 0, 1)
            signal = Signal(rr, annotations)
            runs = Runs(signal)
            #decc_runs = runs.count_for_all(">")
            #acc_runs = runs.count_for_all("<")
            #neutral_runs = runs.count_for_all("==")
            #dec_entropy = runs.HDR
            #assert dec_entropy == pytest.approx(truth["HDR"], 5)
            #acc_entropy = runs.HAR
            #assert acc_entropy == pytest.approx(truth["HAR"],5)
            neutral_entropy = runs.HNO
            assert neutral_entropy == pytest.approx(truth["HNO"], 5)

if __name__ == '__main__':
    unittest.main()
