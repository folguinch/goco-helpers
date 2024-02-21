"""Tools for running the `tclean` CASA task."""
#from typing import Any, Callable, List, Optional, Sequence, TypeVar, Mapping
from typing import Callable, Sequence
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess

from astropy.io import fits
from astropy.stats import mad_std
from casatasks import tclean, exportfits #, imsubimage
import astropy.units as u
#from casatools import image
#import numpy as np

from .common_types import SectionProxy
#from .data_handler import DataHandler
from .utils import get_func_params#, iter_data

def get_tclean_params(
    config: SectionProxy,
    required_keys: Sequence[str] = ('cell', 'imsize'),
    ignore_keys: Sequence[str] = ('vis', 'imagename', 'spw'),
    float_keys: Sequence[str]  = ('robust', 'pblimit', 'pbmask'),
    int_keys: Sequence[str] = ('niter', 'chanchunks'),
    bool_keys: Sequence[str] = ('interactive', 'parallel', 'pbcor',
                                'perchanweightdensity'),
    int_list_keys: Sequence[str] = ('imsize', 'scales'),
) -> dict:
    """Filter input parameters and convert values to the correct type.

    Args:
      config: `ConfigParser` section proxy with input parameters to filter.
      ignore_keys: optional; tclean parameters to ignore.
      float_keys: optional; tclean parameters to convert to float.
      int_keys: optional; tclean parameters to convert to int.
      bool_keys: optional; tclean parameters to convert to bool.
      int_list_keys: optional; tclean parameters as list of integers.
    """
    # Get params
    tclean_pars = get_func_params(tclean, config, required_keys=required_keys,
                                  ignore_keys=ignore_keys,
                                  float_keys=float_keys, int_keys=int_keys,
                                  bool_keys=bool_keys,
                                  int_list_keys=int_list_keys)
    if len(tclean_pars['imsize']) == 1:
        tclean_pars['imsize'] = tclean_pars['imsize'] * 2

    return tclean_pars

def tclean_parallel(vis: Sequence[Path],
                    imagename: Path,
                    nproc: int,
                    tclean_args: dict,
                    log: Callable = print):
    """Run `tclean` in parallel.

    If the number of processes (`nproc`) is 1, then it is run in a single
    processor. The environmental variable `MPICASA` is used to run the code,
    otherwise it will use the `mpicasa` and `casa` available in the system.

    A new logging file is created by `mpicasa`. This is located in the same
    directory where the program is executed.

    Args:
      vis: measurement set.
      imagename: image file name.
      nproc: number of processes.
      tclean_args: other arguments for tclean.
      log: optional; logging function.
    """
    if nproc == 1:
        tclean_args.update({'parallel': False})
        tclean(vis=str(vis), imagename=str(imagename), **tclean_args)
    else:
        # Save tclean params
        tclean_args.update({'parallel': True})
        paramsfile = imagename.parent / 'tclean_params.json'
        paramsfile.write_text(json.dumps(tclean_args, indent=4))

        # Run
        inpvis = ' '.join(map(str, vis))
        cmd = os.environ.get('MPICASA', f'mpicasa -n {nproc} casa')
        script = Path(__file__).parent / 'run_tclean_parallel.py'
        logfile = datetime.now().isoformat(timespec='milliseconds')
        logfile = f'tclean_parallel_{logfile}.log'
        cmd = (f'{cmd} --nogui --logfile {logfile} '
               f'-c {script} {imagename} {paramsfile} {inpvis}')
        log(f'Running: {cmd}')
        # pylint: disable=R1732
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        proc.wait()

def pb_clean(vis: Sequence[Path],
             imagename: Path,
             nproc: int,
             nsigma: float = 3.,
             log: Callable = print,
             **tclean_args: dict):
    """Perform cleaning with a PB mask."""
    # CLEAN args
    pb_clean_args = {'usemask': 'pb',
                     'pbmask': 0.2}
    tclean_args.update(pb_clean_args)
    niter = tclean_args.get('niter', 100000)

    # First pass
    log('Calculating dirty image')
    tclean_args['niter'] = 0
    tclean_parallel(vis, imagename, nproc, tclean_args, log=log)
    clean_imagename = imagename.with_suffix(f'{imagename.suffix}.image')
    fitsimage = clean_imagename.with_suffix(f'.image.fits')
    exportfits(imagename=f'{clean_imagename}', fitsimage=f'{fitsimage}',
               overwrite=True)

    # Get rms
    dirty = fits.open(fitsimage)[0]
    rms = mad_std(dirty.data, ignore_nan=True) * u.Unit(dirty.header['BUNIT'])
    rms = rms.to(u.mJy/u.beam)
    log(f'Dirty image rms: {rms}')
    threshold = nsigma * rms * u.beam

    # Final run
    tclean_args['niter'] = niter
    tclean_args['threshold'] = f'{threshold.value}{threshold.unit}'
    tclean_args['calcres'] = False
    tclean_args['calcpsf'] = False
    tclean_parallel(vis, imagename, nproc, tclean_args, log=log)
    exportfits(imagename=f'{clean_imagename}', fitsimage=f'{fitsimage}',
               overwrite=True)

#def compute_dirty(data: Sequence['DataHandler'],
#                  dirty_dir: Path,
#                  config: SectionProxy,
#                  nproc: int = 4,
#                  redo: bool = False,
#                  log: Callable = print) -> Sequence['DataHandler']:
#    """Compute dirty images.
#
#    It updates the data handler if previously not initiated.
#
#    Args:
#      data: the list containing the data handlers.
#      dirty_dir: directory for the dirty images.
#      config: section proxy with the configuration.
#      nproc: optional; number of processors to use.
#      redo: optional; recompute images if found?
#      log: optional; logging function.
#
#    Returns:
#      An updated `DataHandler`.
#    """
#    # Check what is available
#    content = list(dirty_dir.glob('*.fits'))
#    if len(data) == len(content) == 0:
#        raise ValueError('Cannot run without any data')
#    elif len(data) == 0:
#        log(f'Will use all fits files in {dirty_dir}')
#        data = DataHandler(stems=[fname.stem for fname in content])
#        return [data]
#    else:
#        pass
#
#    # Make the dirty images
#    tclean_pars = get_tclean_params(config)
#    tclean_pars.update({'parallel': nproc > 1,
#                        'niter': 0,
#                        'specmode': 'cube'})
#    for ebdata, spw, stem in iter_data(data):
#        fitsfile = dirty_dir / f'{stem}.image.fits'
#        if fitsfile.is_file() and not redo:
#            log(f'Skipping file {fitsfile}')
#            continue
#        tclean_pars['spw'] = spw
#
#        # Clean data
#        imagename = dirty_dir / stem
#        tclean_parallel(ebdata.uvdata, imagename, nproc, tclean_pars,
#                        log=log)
#        os.system((f'rm -r {imagename}'
#                   '{.model,.sumwt,.pb,.psf,.residual}'))
#        imagename = dirty_dir / f'{stem}.image'
#
#        # Crop data
#        if config.getboolean('crop_images', fallback=False):
#            crop_imagename = imagename.with_suffix('.subimage')
#            box = '{0},{1},{2},{3}'.format(tclean_pars['imsize'][0]/4,
#                                           tclean_pars['imsize'][1]/4,
#                                           tclean_pars['imsize'][0]*3/4,
#                                           tclean_pars['imsize'][1]*3/4)
#            imsubimage(imagename=str(imagename),
#                       outfile=str(crop_imagename),
#                       box=box)
#            os.system(f'rm -r {imagename}')
#            imagename = crop_imagename
#
#        # Export FITS
#        exportfits(imagename=str(imagename), fitsimage=str(fitsfile),
#                   overwrite=redo)
#
#        # Leave only FITS
#        os.system(f'rm -r {imagename}')
#
#    return data
