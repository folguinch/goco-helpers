"""Interferometry tools.

Includes a calculator to compute:

- Bin sizes for channel averaging
- Pixel sizes
- Image sizes
"""
from typing import Optional, List, Callable, Tuple, Union, Dict
from datetime import datetime
import argparse
import sys

from casatools import ms, synthesisutils, msmetadata
import astropy.units as u
import numpy as np
try:
    import analysisUtils as au
except ImportError:
    au = None

import goco_helpers.argparse_actions as actions
import goco_helpers.argparse_parents as parents

def get_fields(msname: 'pathlib.Path',
               intent: str = 'OBSERVE_TARGET#ON_SOURCE'):
    """Get fields from MS."""
    metadata = msmetadata()
    metadata.open(f'{msname}')
    fields = metadata.fieldsforintent(intent, asnames=True)
    metadata.close()

    return fields

def spws_for_names(msname: 'pathlib.Path') -> List[List[int]]:
    """SPW indices per each SPW name."""
    # Get names
    metadata = msmetadata()
    metadata.open(f'{msname}')
    names = metadata.spwsfornames()
    metadata.close()

    # Organize
    organized = {}
    for name, vals in names.items():
        # Extract information
        units = name.split('#')
        if 'BB' not in units[-3]:
            raise NotImplementedError('SPW name format not implemented: BB')
        if 'SW' not in units[-2]:
            raise NotImplementedError('SPW name format not implemented: SW')
        baseband = int(units[-3].split('_')[-1])
        spw = int(units[-2].split('-')[-1])

        # Record spws
        key = (baseband, spw)
        if key in organized:
            organized[key] = organized[key].append(vals)
        else:
            organized[key] = vals

    # Sort keys
    organized = dict(sorted(organized.items(), key=lambda x: x[0]))

    return list(organized.values())

def spws_per_eb(msname: 'pathlib.Path'):
    """Determine the spws for each EB in input MS."""
    spws = {}
    names = spws_for_names(msname)
    for val in names:
        for i, spw in enumerate(val):
            try:
                spws[i+1].append(spw)
            except KeyError:
                spws[i+1] =  [spw]

    return spws

def flag_freqs_to_channels(spw: int,
                           flags: List[Tuple[u.Quantity]],
                           uvdata: 'pathlib.Path',
                           invert: bool = False) -> str:
    """Convert flags in LSRK frequencies to channels."""
    # Open with MS tool
    mstool = ms()
    mstool.open(f'{uvdata}')

    # MS frequency axis to masked array
    freqs = mstool.cvelfreqs(spwids=[int(spw)], outframe='LSRK') * u.Hz
    freqs = np.ma.array(freqs.to(u.GHz).value)
    mstool.close()

    # Iterate over ranges
    for freq_range in flags:
        freq_low = min(freq_range).to(u.GHz).value
        freq_high = max(freq_range).to(u.GHz).value
        freqs = np.ma.masked_where((freqs>=freq_low) & (freqs<=freq_high),
                                   freqs)

    # Convert to indices
    if invert:
        clumps = np.ma.clump_unmasked(freqs)
    else:
        clumps = np.ma.clump_masked(freqs)
    spw_flags = []
    for clump in clumps:
        spw_flags.append(f'{clump.start}~{clump.stop-1}')

    return ';'.join(spw_flags)

def uvdata_info(msname: 'pathlib.Path', log: Callable = print):
    """Extract information from input uv data."""
    # Open MS
    log(f'Opening MS: {msname}')
    mstool = ms()
    mstool.open(f'{msname}')
    metadata = mstool.metadata()

    # Get information
    info = mstool.range(['chan_freq', 'uvdist'])
    max_baseline = np.max(info['uvdist']) * u.m
    log(f'Maximum baseline: {max_baseline:.0f}')

    # Antenna diameters
    is12m = all(int(val['value']) == 12
                for val in metadata.antennadiameter().values())
    diameter = 12 * u.m if is12m else 7 * u.m
    log(f'Antennae diameter: {diameter}')

    # Info per spw
    nspw = metadata.nspw()
    bandwidths = metadata.bandwidths()
    log(f'Number of spectral windows: {nspw}')
    log('Spectral windows:')
    freq_ranges = {}
    for i in range(nspw):
        chan_freqs = mstool.cvelfreqs(spwids=[i], outframe='LSRK')
        #chan_freqs = metadata.chanfreqs(i)
        freq_ranges[i] = {'min': np.min(chan_freqs) * u.Hz,
                          'max': np.max(chan_freqs) * u.Hz,
                          'bandwidth': bandwidths[i] * u.Hz,
                          'nchans': metadata.nchan(i)}
        log((f"{i}: {freq_ranges[i]['min'].to(u.GHz):.4f} - "
             f"{freq_ranges[i]['max'].to(u.GHz):.4f} "
             f"({freq_ranges[i]['bandwidth'].to(u.GHz):.3f} - "
             f"{freq_ranges[i]['nchans']} chans)"))

    # Close MS
    mstool.close()
    return {'max_baseline': max_baseline, 'diameter': diameter,
            'spw_info': freq_ranges}

def max_chan_width(freq: u.Quantity,
                   diameter: u.Quantity,
                   max_baseline: u.Quantity,
                   reduction: float = 0.99) -> u.Quantity:
    """Calculate the maximum channel width.

    Given the desired reduction in peak response value, it calculates the
    maximum channel width accepted. The maximum channel width is calculated by
    solving the equations in [CASA guides]
    (https://safe.nrao.edu/wiki/pub/Main/RadioTutorial/BandwidthSmearing.pdf).

    Args:
      freq: frequency.
      diameter: antenna diameter.
      max_baseline: maximum baseline.
      reduction: optional; reduction in peak response.

    Returns:
      Maximum channel width using a Gaussian response.
      Maximum channel width using a square bandpass.
    """
    chan_width = (freq * 2 * np.sqrt(np.log(2)) * diameter / max_baseline *
                  np.sqrt(1/reduction**2 - 1))

    return chan_width.to(u.MHz), 1.46664 * chan_width.to(u.MHz)

def find_near_exact_denominator(num: int, den: int,
                                direction: str = 'up') -> Tuple[int, int]:
    """Find a closer common denominator to the input one."""
    if num % den == 0:
        return den, num//den
    else:
        if direction == 'up':
            inc = 1
        else:
            inc = -1
        return find_near_exact_denominator(num, den + inc, direction=direction)

def gaussian_primary_beam(freq: u.Quantity, diameter: u.Quantity) -> u.Quantity:
    """Calculate the angular size of a Gaussian primary beam."""
    b = 1 / np.sqrt(np.log(2))
    wvl = freq.to(u.m, equivalencies=u.spectral())
    pb = b * wvl / diameter * u.rad
    return pb.to(u.arcsec)

def gaussian_beam(freq: u.Quantity, baseline: u.Quantity) -> u.Quantity:
    """Estimate the synthetize Gaussian beam size."""
    wvl = freq.to(u.m, equivalencies=u.spectral())
    beam = wvl / baseline * u.rad

    return beam.to(u.arcsec)

def round_sigfig(val: Union[u.Quantity, float],
                 sigfig: int = 1) -> Union[u.Quantity, float]:
    """Round number to given number of significant figures."""
    #logval = np.rint(np.log10(val.value))
    #newval =  np.round(val.value / 10**logval, sigfig-1) * 10**logval
    newval = float(f'{val.value:.{sigfig}g}')

    return newval * val.unit

def baseline_percentile(msname: 'pathlib.Path',
                        percentile: float) -> float:
    """Get the baseline length at `percentile`."""
    if au is None:
        return None
    vals = au.getBaselineStats(f'{msname}', percentile=percentile)

    return vals[0] * u.m

def get_widths(msname: Optional['pathlib.Path'] = None,
               reduction: float = 0.99,
               log: Callable = print,
               **ms_info) -> List[int]:
    """Calculate optimal widths for channel binning.
    
    Values accepted keyword arguments:
    - `diameter`: antenna diameter.
    - `max_baseline`: longest baseline.
    - `spw_info`: frequency `min` and `max`, and `nchans` per SPW.

    Args:
      msname: optional; MS file name.
      reduction: optional; reduction in peak response.
      log: optional; logging function.
      ms_info: optional; parameters in case `msname` is not given.
    """
    # Get info value
    if msname is not None:
        ms_info = uvdata_info(msname, log=log)

    # Extract some quantities
    diameter = ms_info['diameter']
    max_baseline = ms_info['max_baseline']

    # Iterate over spws
    log(f'Using reduction in peak response: {reduction}')
    widths = []
    for spw, info in ms_info['spw_info'].items():
        # Get values
        max_width_min = max_chan_width(info['min'], diameter, max_baseline,
                                       reduction=reduction)
        max_width_max = max_chan_width(info['max'], diameter, max_baseline,
                                       reduction=reduction)

        # Log
        print('-' * 80)
        log(f'SPW: {spw}')
        log(f'Range: {max_width_min[0]:.3f} - {max_width_max[0]:.3f}')

        # Find number of channels per bin
        max_width = min(max_width_min[0], max_width_max[0])
        ngroups = info['bandwidth'] / max_width
        ngroups = int(np.ceil(ngroups.to(1).value))
        if ngroups <= 1:
            binsize = info['nchans']
        else:
            ngroups, binsize = find_near_exact_denominator(info['nchans'],
                                                           ngroups)
        log(f'Minumum value of maximum widths: {max_width:.3f}')
        log((f"SPW bandwidth: {info['bandwidth']} "
             f' ({ngroups} groups of {binsize} channels)'))

        # Store value
        widths.append(binsize)

    return widths

def imaging_parameters(msname: Optional['pathlib.Path'] = None,
                       oversample: float = 5.,
                       pbcoverage: float = 2.,
                       log: Callable = print,
                       **ms_info) -> Dict:
    """Determine imaging parameters from uv data.

    Values accepted for keyword arguments:
    - `diameter`: antenna diameter.
    - `max_baseline`: longest baseline.
    - `spw_info`: frequency `min` and `max` per SPW.

    Args:
      msname: Optional. MS file name.
      oversample: Optional. number of pixels per beam.
      pbcoverage: Optional. image size in terms of primary beam size.
      log: Optional. logging function.
      ms_info: Optional. parameters in case `msname` is not given.
    """
    # Get info value
    if msname is not None:
        ms_info = uvdata_info(msname, log=log)

    # Get some values
    su = synthesisutils()
    diameter = ms_info['diameter']
    max_baseline = ms_info['max_baseline']

    # Iterate over spws
    log(f'Using beam overample: {oversample}')
    params = {'cell': np.inf * u.arcsec, 'imsize': 0}
    spws = []
    for spw, info in ms_info['spw_info'].items():
        # Get values
        beam = round_sigfig(gaussian_beam(info['max'], max_baseline))
        pbsize = np.round(gaussian_primary_beam(info['min'], diameter), 1)
        pixsize = beam / oversample
        imsize = int(pbsize * pbcoverage / pixsize)

        # Optimal size
        imsize_opt = su.getOptimumSize(imsize)

        # Log
        print('-' * 80)
        log(f'SPW: {spw}')
        log(f'Estimated highest resolution: {beam}')
        log(f'Estimated largest pb size: {pbsize}')
        log(f'Estimated pixel size: {pixsize}')
        log(f'Estimated imsize: {imsize}')
        log(f'Optimal estimated imsize: {imsize_opt}')

        # Update params
        params['cell'] = min(params['cell'], pixsize)
        params['imsize'] = max(params['imsize'], imsize_opt)
        spws.append(f'{spw}')
    params['spws'] = ','.join(spws)

    # Get b75
    b75 = baseline_percentile(msname, 75)
    if b75 is not None:
        params['b75'] = f'{b75.value:.1f} {b75.unit}'

    return params

def _get_uvdata_info(args: argparse.Namespace):
    args.info = uvdata_info(args.uvdata[0], log=args.log.info)

def _max_chan_width(args: argparse.Namespace):
    get_widths(reduction=args.reduction[0],
               log=args.log.info,
               **args.info)

def _imaging_parameters(args: argparse.Namespace):
    imaging_parameters(oversample=args.oversample[0],
                       pbcoverage=args.pbcoverage[0],
                       log=args.log.info,
                       **args.info)

def calculator(args: Optional[List] = None) -> None:
    # Steps
    steps = [_get_uvdata_info, _max_chan_width, _imaging_parameters]

    # Command line options
    logfile = datetime.now().isoformat(timespec='milliseconds')
    logfile = f'debug_mstools_{logfile}.log'
    args_parents = [parents.logger(logfile)]
    parser = argparse.ArgumentParser(
        description='Imaging calculator',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents,
        conflict_handler='resolve',
    )
    parser.add_argument('--oversample', nargs=1, type=float,
                        default=[5.],
                        help='Number of pixels per beam')
    parser.add_argument('--pbcoverage', nargs=1, type=float,
                        default=[2.],
                        help='Image size in terms of primary beam size')
    parser.add_argument('-r', '--reduction', nargs=1, type=float,
                        default=[0.99],
                        help='reduction in the peak response')
    parser.add_argument('uvdata', nargs=1, action=actions.CheckDir,
                        help='uv data MS name')
    parser.set_defaults(info=None)

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    for step in steps:
        step(args)

if __name__ == '__main__':
    calculator(sys.argv[1:])
