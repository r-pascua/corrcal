# corrcal

[high-level description]

## Installation

Basic installation instructions, with all of the dependencies. Note
that this is designed for use on Linux distributions and has not been
tested on Windows or MacOS. First, clone the repository via
```
git clone https://github.com/r-pascua/corrcal.git
```
and move into the new `corrcal` directory. To create a fresh environment
for e.g. developing, do
```
conda env create -f ci/tests.yaml [-n env_name]
pip install [-e] .
```
The `-n` flag should come *after* the configuration file specification.
The `-e` flag should be used if you intend to do any development work.
