# corrcal

The `corrcal` package provides low-level interfaces necessary for implementing
***Corr***elation ***Cal***ibration. CorrCal is a *covariance* optimization scheme
for radio interferometric arrays that simulatneously leverages the strengths of
sky-based and redundant calibration while being fairly insensitive to errors in
the model visibility-visibility covariance. Although CorrCal relies on optimally
matching the model covariance to the data covariance, it scales linearly with the
number of baselines used for calibration due to the sparsity of the model covariance.

The routines provided in this package are described in detail in 
[Pascua, Sievers, & Liu (2026)](https://arxiv.org/abs/2602.06109), which also
details several validation tests carried out to assess the performance of CorrCal.

## Package Details


## Installation

This package has been developed for use on Unix-like machines, and has not
been tested on Windows machines. The simplest way to install `corrcal` is by
installing the latest released version via `pip install corrcal`. You may
alternatively clone the repository, navigate to the local clone, and install
from the top-level directory:

```
git clone https://github.com/r-pascua/corrcal.git
cd corrcal
pip install [-e] .  # Set the -e flag for an editable installation.
```

## Dependencies
- python>3.10
- numpy
- scipy

### Requirements for Model Building Utilities
- astropy
- pyuvdata
- pyradiosky
- healpy
