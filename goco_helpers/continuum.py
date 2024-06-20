#!python3
"""Calculate a quick continuum visibility and images."""
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import argparse
import os
import sys

from casaplotms import plotms
from casatasks import flagdata, flagmanager, split
import astropy.units as u
import numpy as np

from goco_helpers.clean_tasks import pb_clean
from goco_helpers.common_types import SectionProxy
from goco_helpers.mstools import get_widths, imaging_parameters
import goco_helpers.argparse_actions as actions
import goco_helpers.argparse_parents as parents

def get_continuum(msname: 'pathlib.Path',
                  output: 'pathlib.Path',
                  config: Optional[SectionProxy] = None,
                  flags: Optional[str] = None,
                  plotdir: Optional['pathlib.Path'] = None,
                  **plotargs):
    """Calculate continuum visibilities.

    Args:
      msname: target uv data.
      output: output file name.
      config: optional; parameters for `split` task.
      flags: optional; frequency ranges to flag.
    """
    # Prepare ms
    if flags is not None:
        # Clean previous flag versions
        flag_list = flagmanager(vis=f'{msname}', mode='list')
        for flag_ver in flag_list.values():
            try:
                if 'before_cont_flags' in list(flag_ver.values()):
                    flagmanager(vis=f'{msname}', mode='remove',
                                versionname='before_cont_flags')
                elif 'afoli_flags' in list(flag_ver.values()):
                    flagmanager(vis=f'{msname}', mode='remove',
                                versionname='before_cont_flags')
            except AttributeError:
                continue

        # Flag data
        flagmanager(vis=f'{msname}', mode='save',
                    versionname='before_cont_flags')
        flagdata(vis=f'{msname}', mode='manual', spw=flags,
                 flagbackup=False)
        flagmanager(vis=f'{msname}', mode='save',
                    versionname='afoli_flags')

    # Widths
    if config is not None:
        width = list(map(int, config['width'].split(',')))
        datacolumn = config['datacolumn']
    else:
        width = get_widths(msname)
        datacolumn = 'data'

    # Plot MSs
    if plotdir is not None and plotms is not None:
        plotfile = plotdir / msname.with_suffix('.png').name
        if flags is not None:
            plotfile = plotfile.with_suffix('.flagged.png')
        plotms(vis=f'{msname}', xaxis='freq', yaxis='amplitude',
               title='Not averaged data', showgui=False,
               overwrite=True, plotfile=f'{plotfile}',
               **plotargs)

    # Split data
    split(vis=f'{msname}', outputvis=f'{output}', width=width,
          datacolumn=datacolumn)

    # Plot MSs
    if plotdir is not None and plotms is not None:
        plotfile = plotdir / output.with_suffix('.png').name
        plotms(vis=f'{output}', xaxis='freq', yaxis='amplitude',
               title='Averaged data', showgui=False,
               overwrite=True, plotfile=f'{plotfile}',
               **plotargs)

    # Restore flags
    if flags is not None:
        flagmanager(vis=f'{msname}', mode='restore',
                    versionname='before_cont_flags')

def _get_continuum(args: argparse.Namespace):
    outmss = []
    for uvdata in args.uvdata:
        if args.outdir is not None:
            outdir = args.outdir[0]
        else:
            outdir = uvdata.parent
        name = uvdata.with_suffix('.quick_cont.ms').name
        output = outdir / name
        if output.exists() and args.resume:
            outmss.append(output)
            continue
        elif output.exists() and not args.resume:
            args.log.info('Deleting quick continuum MS')
            os.system(f'rm -rf {output}')
        get_continuum(uvdata, output)
        outmss.append(output)

    return outmss

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
                        default=[Path('./')],
                        help='Output directory')
    parser.add_argument('uvdata', nargs='*', action=actions.CheckDir,
                        help='uv data MS name')
    #parser.set_defaults()

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    msnames = _get_continuum(args)
    imagename = msnames[0].with_suffix('.pbclean')
    imagename = args.outdir[0] / imagename.name
    tclean_pars = {'cell': np.inf * u.arcsec,
                   'imsize': 0}
    for uvdata in args.uvdata:
        aux = imaging_parameters(uvdata, log=args.log.info)
        tclean_pars['cell'] = min(aux['cell'], tclean_pars['cell'])
        tclean_pars['imsize'] = max(aux['imsize'], tclean_pars['imsize'])
    tclean_pars['cell'] = (f"{tclean_pars['cell'].value}"
                           f"{tclean_pars['cell'].unit}")
    tclean_pars['specmode'] = 'mfs'
    tclean_pars['outframe'] = 'LSRK'
    tclean_pars['gridder'] = 'standard'
    tclean_pars['deconvolver'] = 'hogbom'
    tclean_pars['weighting'] = 'briggs'
    tclean_pars['robust'] = 0.5
    pb_clean(msnames, imagename, args.nproc[0], log=args.log.info, **tclean_pars)

if __name__ == '__main__':
    quick_continuum(sys.argv[1:])
