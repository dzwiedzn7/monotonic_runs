import pytest
import os
import pandas as pd

@pytest.fixture
def resource_files():
    resource_dir = "resources/rr"
    file_paths = []

    for root, dirs, files in os.walk(resource_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths

@pytest.fixture
def ground_truth_df():
    return pd.read_excel("resources/ground_truth.xlsx")