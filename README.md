# Helpers for go-continuum

A set of tools for [`go-continuum`](https://github.com/folguinch/go-continuum)
but that can be useful in other projects.

## Installation

Using `pip`:
```bash
pip install git+https://github.com/folguinch/goco-helpers
```

### Developer installation

The preferred method is through [`poetry`](https://python-poetry.org/):
```bash
git clone git@github.com:folguinch/goco-helpers.git
cd goco-helpers
poetry install
```

It can also be installed through `pip`:
```bash
git clone git@github.com:folguinch/goco-helpers.git
cd goco-helpers
pip install -e .
```

## Generating configuration files

Configuration files are needed for `go-continuum` to operate. The script
`config_generator.py` provide an easy way to generate configuration files
for as many sources as needed.

To generate configuration files for several fields with their own MSs:
```bash
python -m goco_helpers.config_generator --outdir output_directory defaults.cfg uvdata1.ms uvdata2.ms ...
```
A template example can be found in the `examples` directory, and contains all
the available options to set default values. All the needed parameters that are
not in the default file are determined from the `uvdata` and filled by the
script.

To determine the 75 percentile baseline for `auto-masking`,
[Analysis Utilities](https://casaguides.nrao.edu/index.php/Analysis_Utilities)
can be imported if available in the `PYTHONPATH` environmental variable.
