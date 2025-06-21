# Monotonic Runs

## Description
This repository contains Python code for implementing the Monotonic Runs method, a time series analysis technique. This method divides a signal into series of "runs" where the signal monotonically accelerates, decelerates, or remains neutral. The Monotonic Runs method has been proven reliable for assessing asymmetrical properties of signals, particularly in the field of heart rate variability analysis.

For more information on the method, refer to the article:
[DOI: 10.1088/0967-3334/32/8/002](https://doi.org/10.1088/0967-3334/32/8/002)

## Technologies
- Python
- NumPy

## Installation
You can install this package using one of the following methods:

1. Using setuptools:
   ```
   python setup.py install
   ```

2. Using requirements.txt:
   ```
   pip install -r requirements.txt
   ```

3. Direct use:
   You can directly use the main script located at `main/src/core/runs/runs_entropy.py`

## Usage
Here's an example of how to use the package to calculate Shannon entropy for every type of run in a chi-square distributed random signal:

```python
from monotonic_runs import Signal, Runs
import numpy as np

# Assuming chi_square is your input signal
chi_square = np.random.chisquare(df=2, size=1000)

signal = Signal(chi_square, np.zeros_like(chi_square))
runs = Runs(signal)

dec_entropy = runs.HDR
acc_entropy = runs.HAR
neutral_entropy = runs.HNO

print(dec_entropy, acc_entropy, neutral_entropy)
```

## Features
- Divide time series into monotonic runs
- Calculate Shannon entropy for different types of runs (accelerating, decelerating, neutral)
- Assess asymmetrical properties of signals

## Contributing
Contributions to improve the Monotonic Runs method implementation are welcome. Please feel free to submit a Pull Request.

## License
GPL-3.0

## Contact
bbiczuk@gmail.com
