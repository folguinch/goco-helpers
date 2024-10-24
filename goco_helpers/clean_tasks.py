"""Tools for running the `tclean` CASA task."""
from typing import Callable, Sequence, Dict, Tuple, Optional
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
import numpy as np

from .common_types import SectionProxy
#from .data_handler import DataHandler
from .utils import get_func_params#, iter_data

def image_sn(fitsimage: Path,
             default_unit: u.Unit = u.mJy/u.beam
             ) -> Tuple[u.Quantity, u.Quantity]:
    """Calculate the peak and noise of an image."""
    image = fits.open(fitsimage)[0]
    unit = u.Unit(image.header['BUNIT'])
    peak = np.nanmax(image.data) * unit
    peak = peak.to(default_unit)
    rms = mad_std(image.data, ignore_nan=True) * unit
    rms = rms.to(default_unit)

    return peak, rms

def get_tclean_params(
    config: SectionProxy,
    required_keys: Sequence[str] = ('cell', 'imsize'),
    ignore_keys: Sequence[str] = ('vis', 'imagename', 'spw', 'nsigma'),
    float_keys: Sequence[str]  = ('robust', 'pblimit', 'pbmask',
                                  'noisethreshold', 'sidelobethreshold',
                                  'lownoisethreshold', 'minbeamfrac',
                                  'negativethreshold'),
    int_keys: Sequence[str] = ('niter', 'chanchunks'),
    bool_keys: Sequence[str] = ('interactive', 'parallel', 'pbcor',
                                'perchanweightdensity', 'fastnoise'),
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

def automasking_params(b75: u.Quantity,
                       is_cube: bool,
                       is_aca: bool = False,
                       is_combined: bool = False) -> Dict:
    """Get recommended auto-masking parameters.

    Parameters are taken from the table in the [CASA guides](
    https://casaguides.nrao.edu/index.php/Automasking_Guide_CASA_6.6.1)

    Args:
      b75: Baseline at 75th percentile.
      is_cube: Is imaging for cubes?
      is_aca: Optional. Is the data from ACA?
      is_combined: Optional. Is the data from 12m and 7m arrays?

    Returns:
      A dictionary with the parameter values.
    """
    if is_aca:
        parameters = {'noisethreshold': 5.0,
                      'sidelobethreshold': 1.25,
                      'lownoisethreshold': 2.0,
                      'minbeamfrac': 0.1,
                      'negativethreshold': 0.0,
                      'fastnoise': False}
    elif is_combined:
        parameters = {'noisethreshold': 4.25,
                      'sidelobethreshold': 2.0,
                      'lownoisethreshold': 1.5,
                      'minbeamfrac': 0.3,
                      'negativethreshold': 0.0,
                      'fastnoise': False}
    else:
        if b75 < 300 * u.m:
            negativethreshold = 15. if is_cube else 0.
            parameters = {'noisethreshold': 4.25,
                          'sidelobethreshold': 2.0,
                          'lownoisethreshold': 1.5,
                          'minbeamfrac': 0.3,
                          'negativethreshold': negativethreshold,
                          'fastnoise': False}
        elif b75 < 400 * u.m:
            negativethreshold = 7. if is_cube else 0.
            parameters = {'noisethreshold': 5.0,
                          'sidelobethreshold': 2.0,
                          'lownoisethreshold': 1.5,
                          'minbeamfrac': 0.3,
                          'negativethreshold': negativethreshold,
                          'fastnoise': False}
        else:
            negativethreshold = 7. if is_cube else 0.
            parameters = {'noisethreshold': 5.0,
                          'sidelobethreshold': 2.5,
                          'lownoisethreshold': 1.5,
                          'minbeamfrac': 0.3,
                          'negativethreshold': negativethreshold,
                          'fastnoise': True}
    return parameters

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
      vis: measurement sets.
      imagename: image file name.
      nproc: number of processes.
      tclean_args: other arguments for tclean.
      log: optional; logging function.
    """
    if nproc == 1:
        tclean_args.update({'parallel': False})
        info = tclean(vis=list(map(str, vis)), imagename=str(imagename),
                      **tclean_args)
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
             nproc: int = 1,
             nsigma: float = 3.,
             thresh_niter: float = 2,
             log: Callable = print,
             pbmask: float = 0.2,
             **tclean_args: Dict) -> Dict:
    """Perform cleaning with a PB mask.

    Args:
      vis: List of visibilities.
      imagename: Base name for image name.
      nproc: Optional. Number of parallel processes.
      nsigma: Optional. Noise (rms) level for the threshold.
      thresh_niter: Optional. Number of iteration to perform.
      log: Optional. Logging function.
      pbmask: Optional. Mask PB limit.
      tclean_args: Optional. Additional arguments for `tclean`.

    Returns:
      A dictionary with the FITS file name and last threshold.
    """
    pb_clean_args = {'usemask': 'pb',
                     'pbmask': pbmask}
    tclean_args.update(pb_clean_args)
    info = iter_clean(vis, imagename, nproc=nproc, nsigma=nsigma,
                      thresh_niter=thresh_niter, log=log, **tclean_args)

    return info

def auto_masking(vis: Sequence[Path],
                 imagename: Path,
                 nproc: int = 1,
                 b75: Optional[u.Quantity] = None,
                 is_aca: bool = False,
                 is_combined: bool = False,
                 nsigma: float = 3.,
                 thresh_niter: float = 2,
                 log: Callable = print,
                 **tclean_args: dict) -> Dict:
    """Clean images iteratively using CASA auto-masking.

    The `b75` parameter (75th percentile baseline) is required to determine the
    automasking parameters from the recommended values in the [CASA guides](
    https://casaguides.nrao.edu/index.php/Automasking_Guide_CASA_6.6.1). If not
    given the `tclean` defaults are used or the values given in `tclean_args`.

    Args:
      vis: List of visibilities.
      imagename: Base name for image name.
      nproc: Optional. Number of parallel processes.
      b75: Optional. Baseline at 75th percentile.
      is_aca: Optional. Is the data from ACA?
      is_combined: Optional. Is the data from the 12m and 7m arrays?
      nsigma: Optional. Noise (rms) level for the threshold.
      thresh_niter: Optional. Number of iteration to perform.
      log: Optional. Logging function.
      tclean_args: Optional. Additional arguments for `tclean`.

    Returns:
      A dictionary with the FITS file name and last threshold.
    """
    # Get auto-masking parameters
    clean_args = {'usemask': 'auto-multithresh'}
    if b75 is not None:
        is_cube = 'cube' in tclean_args.get('specmode', 'mfs')
        clean_args.update(automasking_params(b75, is_cube, is_aca=is_aca,
                                             is_combined=is_combined))

    # Clean iteratively
    tclean_args.update(clean_args)
    info = iter_clean(vis, imagename, nproc=nproc, nsigma=nsigma,
                      thresh_niter=thresh_niter, log=log, **tclean_args)

    return info

def iter_clean(vis: Sequence[Path],
               imagename: Path,
               nproc: int = 1,
               nsigma: float = 3.,
               thresh_niter: float = 2,
               log: Callable = print,
               **tclean_args: dict) -> Dict:
    """Clean visibilities iteratively.

    After each iteration the threshold is recalculated based on the current rms
    level.

    Args:
      vis: List of visibilities.
      imagename: Base name for image name.
      nproc: Optional. Number of parallel processes.
      nsigma: Optional. Noise (rms) level for the threshold.
      thresh_niter: Optional. Number of iteration to perform.
      log: Optional. Logging function.
      tclean_args: Optional. Additional arguments for `tclean`.

    Returns:
      A dictionary with the final image file name (`fitsimage`), threshold for
      each step (`thresholds`).
    """
    # CLEAN args
    niter = tclean_args.get('niter', 100000)
    if tclean_args.get('deconvolver', 'hogbom') == 'mtmfs':
        clean_imagename = imagename.with_suffix(f'{imagename.suffix}.image.tt0')
    else:
        clean_imagename = imagename.with_suffix(f'{imagename.suffix}.image')
    thresholds = []

    # Iterate
    for i in range(thresh_niter):
        # First pass
        if i == 0:
            log('Calculating dirty image')
            tclean_args['niter'] = 0
        else:
            tclean_args['niter'] = niter
            tclean_args['calcres'] = False
            tclean_args['calcpsf'] = False
        tclean_parallel(vis, imagename, nproc, tclean_args, log=log)
        fitsimage = clean_imagename.with_suffix(f'.iter{i}.image.fits')
        exportfits(imagename=f'{clean_imagename}', fitsimage=f'{fitsimage}',
                   overwrite=True)

        # Get rms
        _, rms = image_sn(fitsimage)
        thresholds.append(nsigma * rms * u.beam)
        log((f'Threshold from iteration {i}: '
             f'{nsigma}x{rms * u.beam} = {thresholds[i]}'))

        # Check threshold divergence
        if i > 0 and thresholds[i] > thresholds[i-1]:
            log('Thresholds diverging, CLEAN finished')
            final_fitsimage = clean_imagename.with_suffix('.final.image.fits')
            os.system('mv {fitsimage} {final_fitsimage}')
            return {'fitsimage': final_fitsimage, 'thresholds': thresholds}

        # Set threshold
        tclean_args['threshold'] = f'{thresholds[i].value}{thresholds[i].unit}'

    # Final run
    tclean_args['niter'] = niter
    tclean_parallel(vis, imagename, nproc, tclean_args, log=log)
    fitsimage = clean_imagename.with_suffix('.final.image.fits')
    exportfits(imagename=f'{clean_imagename}', fitsimage=f'{fitsimage}',
               overwrite=True)

    # Pbcor image to fits
    if tclean_args.get('pbcor', False):
        pbcorimage = clean_imagename.with_suffix(
            f'{clean_imagename.suffix}.pbcor')
        pbcor_fits_image = pbcorimage.with_suffix('.final.image.pbcor.fits')
        exportfits(imagename=f'{pbcorimage}', fitsimage=f'{pbcor_fits_image}',
                   overwrite=True)

    return {'fitsimage': fitsimage, 'thresholds': thresholds}

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
