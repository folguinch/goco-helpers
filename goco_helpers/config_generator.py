"""Configuration file generator and tools."""
from typing import Callable, Optional, List, Dict
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
import argparse
import sys

from goco_helpers.mstools import get_fields, imaging_parameters, get_widths
import goco_helpers.argparse_actions as actions
import goco_helpers.argparse_parents as parents

def read_config(filename: Path,
                template: Optional[Path] = None,
                defaults: Optional[Dict] = None) -> ConfigParser:
    """Read configuration file."""
    config = ConfigParser(defaults=defaults)
    if template is not None:
        config.read(template)
    config.read(filename)

    return config

def generate_config_dict(msname: Path,
                         name: str,
                         field: str,
                         log: Callable = print) -> Dict:
    """Generate a configuration dictionary.

    The task generates/modify 3 sections:

    - `DEFAULT`: sets the `name`, `field` and `neb` values.
    - `uvdata`: sets the `original` and `concat` values.
    - `imaging`: sets the `cell` and `imsize` values.
    - `continuum`: sets the `width` value for binning.
    
    Args:
      msname: uv data to extract imaging properties.
      field: field name.
      log: optional; logging function.
    """
    # Create config and update field
    config = {'DEFAULT': {'name': name,
                          'field': field,
                          'neb': 1},
              'uvdata': {'original': [msname],
                         'concat': f'{name}.ms'}}

    # Set imaging values
    log('Setting imaging parameters')
    config['imaging'] = imaging_parameters(msname, log=log)
    print('=' * 80)

    # Set continuum values
    log('Setting continuum parameters')
    config['continuum'] = {'width': get_widths(msname, log=log)}
    print('=' * 80)

    return config

def update_config_dict(config: Dict,
                       msname: Path,
                       log: Callable = print) -> Dict:
    """Update configuration dictionary with values from uv data."""
    # Increase number of EBs
    config['DEFAULT']['neb'] += 1
    config['uvdata']['original'].append(msname)

    # Get new imaging parameters
    log('Getting new imaging parameters')
    img_pars = imaging_parameters(msname, log=log)
    config['imaging']['cell'] = min(img_pars['cell'],
                                    config['imaging']['cell'])
    config['imaging']['imsize'] = max(img_pars['imsize'],
                                      config['imaging']['imsize'])
    print('=' * 80)

    # Get new widths
    log('Concatenating new continuum bins')
    config['continuum']['width'] += get_widths(msname, log=log)
    print('=' * 80)

    return config

def _get_field_data(args: argparse.Namespace):
    field_data = {}
    for msname in args.uvdata:
        fields = set(get_fields(msname))
        args.log.info('Fields: %s', fields)

        for field in fields:
            if field in field_data:
                field_data[field] = update_config_dict(field_data[field],
                                                       msname,
                                                       log=args.log.info)
            else:
                preffix = args.preffix[0]
                name = f'{preffix}{field}'
                field_data[field] = generate_config_dict(msname,
                                                         name,
                                                         field,
                                                         log=args.log.info)

    args.field_data = field_data

def _convert_to_config(args: argparse.Namespace):
    for field, config_dict in args.field_data.items():
        # Convert values to strings
        cell = config_dict['imaging']['cell']
        widths = config_dict['continuum']['width']
        original = config_dict['uvdata']['original']
        config_dict['imaging']['cell'] = f'{cell.value}{cell.unit}'
        config_dict['continuum']['width'] = ','.join(map(str, widths))
        config_dict['uvdata']['original'] = ','.join(map(str, original))

        # Set concat uvdata
        preffix = args.preffix[0]
        suffix = args.suffix[0]
        if len(original) == 1:
            config_dict['uvdata']['concat'] = config_dict['uvdata']['original']
        else:
            config_dict['uvdata']['concat'] = f'./{preffix}{field}{suffix}.ms'

        # Generate parser
        config = ConfigParser()
        config.read(args.template[0])
        config.read_dict(config_dict)

        # Save config
        outdir = args.outdir[0]
        filename = outdir / f'{preffix}{field}{suffix}.cfg'
        with filename.open('w') as fl:
            config.write(fl)

def config_gen(args: Optional[List] = None) -> None:
    """Generate config file from command line arguments."""
    # Steps
    steps = [_get_field_data, _convert_to_config]

    # Command line options
    logfile = datetime.now().isoformat(timespec='milliseconds')
    logfile = f'debug_config_gen_{logfile}.log'
    args_parents = [parents.logger(logfile)]
    parser = argparse.ArgumentParser(
        description='Configuration file generator',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents,
        conflict_handler='resolve',
    )
    parser.add_argument('--outdir', nargs=1, default=[Path('./')],
                        action=actions.CheckDir,
                        help='Output directory')
    parser.add_argument('--preffix', nargs=1, default=[''],
                        help='File name preffix')
    parser.add_argument('--suffix', nargs=1, default=[''],
                        help='File name suffix')
    parser.add_argument('--output', nargs=1, action=actions.NormalizePath,
                        help='Output file name')
    parser.add_argument('template', nargs=1, action=actions.CheckFile,
                        help='Template file name')
    parser.add_argument('uvdata', nargs='+', action=actions.CheckDir,
                        help='uv data MS name')
    parser.set_defaults(field_data=None)

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    for step in steps:
        step(args)

if __name__ == '__main__':
    config_gen(sys.argv[1:])
