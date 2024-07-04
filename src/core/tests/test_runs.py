import unittest
import os
import pytest
import pandas as pd
from src.core.runs.runs_entropy import RRLoader, Signal, Runs
import time


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
        start = time.perf_counter()
        for idx, file_path in enumerate(resource_files):
            truth = ground_truth_df.iloc[idx]
            rr, annotations = RRLoader().load(file_path, 0, 1)
            signal = Signal(rr, annotations)
            runs = Runs(signal)
            dec_entropy = runs.HDR
            acc_entropy = runs.HAR
            neutral_entropy = runs.HNO

            assert dec_entropy == pytest.approx(truth["HDR"], 5)
            assert acc_entropy == pytest.approx(truth["HAR"],5)
            assert neutral_entropy == pytest.approx(truth["HNO"], 5)

        end = time.perf_counter()
        print("Time elapsed: ", end - start)

if __name__ == '__main__':
    unittest.main()
