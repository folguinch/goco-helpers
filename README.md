# GoContinuum helpers

A set of tool for GoContinuum but that can be useful in other projects.

## Generating configuration files

Configuration files are needed for `GoContinuum` to operate. The script
`config_generator.py` provide an easy way to generate configuration files
for as many sources as needed.

To generate configuration files for several fields with their own MSs:
```bash
python -m goco_helpers.config_generator --outdir output_directory defaults.cfg uvdata1.ms uvdata2.ms ...
```
A template example can be found in the `examples` directory, and contains all
the available options to set default values. All the needed parameters that are
not in the default file are determined from the uvdata and filled by the script.
