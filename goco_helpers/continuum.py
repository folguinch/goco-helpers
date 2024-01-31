#!python3
"""Calculate a quick continuum visibility and images."""
from typing import Optional, List
from datetime import datetime
import argparse
import os
import sys

from casatasks import flagdata, flagmanager, split

from goco_helpers.clean_tasks import pb_clean
from goco_helpers.common_types import SectionProxy
from goco_helpers.mstools import get_widths, imaging_parameters
import goco_helpers.argparse_actions as actions
import goco_helpers.argparse_parents as parents

def get_continuum(msname: 'pathlib.Path',
                  output: 'pathlib.Path',
                  config: Optional[SectionProxy] = None,
                  flags: Optional[str] = None):
    """Calculate continuum visibilities.

    Args:
      msname: target uv data.
      output: output file name.
      config: optional; parameters for `split` task.
      flags: optional; frequency ranges to flag.
    """
    # Prepare ms
    if flags is not None:
        flagmanager(vis=f'{msname}', mode='save',
                    versionname='before_cont_flags')
        flagdata(vis=f'{msname}', mode='manual', spw=flags,
                 flagbackup=False)

    # Widths
    if config is not None:
        width = list(map(int, config['width'].split(',')))
        datacolumn = config['datacolumn']
    else:
        width = get_widths(msname)
        datacolumn = 'data'

    # Split data
    split(vis=f'{msname}', outputvis=f'{output}', width=width,
          datacolumn=datacolumn)

    # Restore flags
    if flags is not None:
        flagmanager(vis=f'{msname}', mode='restore',
                    versionname='before_cont_flags')

def _get_continuum(args: argparse.Namespace):
    if args.outdir is not None:
        outdir = args.outdir[0]
    else:
        outdir = args.uvdata[0].parent
    name = args.uvdata[0].with_suffix('.quick_cont.ms').name
    output = outdir / name
    if output.exists() and args.resume:
        return output
    elif output.exists() and not args.resume:
        args.log.info('Deleting quick continuum MS')
        os.system(f'rm -rf {output}')
    get_continuum(args.uvdata[0], output)

    return output

def quick_continuum(args: Optional[List] = None):
    """Compute a quick continuum from command line inputs."""
    # Command line options
    logfile = datetime.now().isoformat(timespec='milliseconds')
    logfile = f'debug_quick_continuum_{logfile}.log'
    args_parents = [parents.logger(logfile)]
    parser = argparse.ArgumentParser(
        description='Quick continuum script',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents,
        conflict_handler='resolve',
    )
    parser.add_argument('--resume', action='store_true',
                        help='Resume from where it was left')
    parser.add_argument('--nproc', nargs=1, type=int, default=[5],
                        help='Number of processes for tclean')
    parser.add_argument('--outdir', nargs=1, action=actions.MakePath,
                        default=None,
                        help='Output directory')
    parser.add_argument('uvdata', nargs=1, action=actions.CheckDir,
                        help='uv data MS name')
    #parser.set_defaults()

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    msname = _get_continuum(args)
    imagename = msname.with_suffix('.pbclean')
    tclean_pars = imaging_parameters(args.uvdata[0], log=args.log.info)
    tclean_pars['cell'] = (f"{tclean_pars['cell'].value}"
                           f"{tclean_pars['cell'].unit}")
    tclean_pars['specmode'] = 'mfs'
    tclean_pars['outframe'] = 'LSRK'
    tclean_pars['gridder'] = 'standard'
    tclean_pars['deconvolver'] = 'hogbom'
    tclean_pars['weighting'] = 'briggs'
    tclean_pars['robust'] = 0.5
    pb_clean(msname, imagename, args.nproc[0], log=args.log.info, **tclean_pars)

if __name__ == '__main__':
    quick_continuum(sys.argv[1:])
