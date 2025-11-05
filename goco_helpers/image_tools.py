"""Tools for working with cubes and 2D images."""
from typing import Optional, Callable, Tuple, Sequence, List

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import numpy.typing as npt

def open_cube(imagename: 'pathlib.Path',
              import_format: Optional[str] = None) -> SpectralCube:
    """Read a cube."""
    if import_format is None:
        if imagename.suffix == '.fits':
            import_format = 'fits'
        else:
            import_format = 'casa'
    image = SpectralCube.read(imagename,
                              use_dask=import_format=='casa',
                              format=import_format)
    image.allow_huge_operations = True
    if import_format == 'casa':
        image.use_dask_scheduler('threads', num_workers=12)

    return image

def combine_positions(positions: List[SkyCoord],
                      method: str = 'mean') -> SkyCoord:
    """Combine position using the requested method."""
    # Set coordinates in same units and system
    coords = SkyCoord([pos.icrs for pos in positions])

    # Combine by method
    if method == 'mean':
        ref = SkyCoord(np.mean(coords.ra), np.mean(coords.dec),
                       frame=coords.frame)
        if len(coords) <= 2:
            return ref
        dist = coords.separation(ref)
        ind = np.nanargmax(dist)
        best_pos = np.delete(coords, ind)
        best_pos = SkyCoord(np.mean(best_pos.ra), np.mean(best_pos.dec),
                            frame=best_pos.frame)
    else:
        raise NotImplementedError(f'Combine method: {method}')

    return best_pos

def sum_collapse(cube: SpectralCube,
                 rms: Optional[u.Quantity] = None,
                 nsigma: float = 5.,
                 edge: int = 10,
                 log: Callable = print) -> fits.PrimaryHDU:
    """Collapse cube along the spectral axis.

    If `rms` is given, then all values over `nsigma*rms` are summed.

    Args:
      cube: Spectral cube.
      rms: Optional. The cube noise level.
      nsigma: Optional. Noise level to filter data.
      edge: Optional. Border channels to ignore.
      log: Optional. Logging function.
    """
    if rms is not None:
        log(f'Summing all values over: {nsigma * rms}')
        masked = cube.with_mask(cube > nsigma * rms)
        masked = masked.with_fill_value(0.)
        imgsum = np.sum(masked.filled_data[edge:-edge, :, :], axis=0)
    else:
        log('Summing along spectral axis')
        imgsum = np.sum(cube.filled_data[edge:-edge, :, :], axis=0)

    return fits.PrimaryHDU(imgsum,
                           header=cube.wcs.sub(['longitude',
                                                'latitude']).to_header())

def max_collapse(cube: SpectralCube,
                 rms: Optional[u.Quantity] = None,
                 nsigma: float = 1.,
                 edge: int = 10,
                 log: Callable = print) -> fits.PrimaryHDU:
    """Collapse cube along the spectral axis using `max` function.

    If `rms` is given, then all values below `nsigma*rms` are set to zero.

    Args:
      cubename: Cube filename.
      rms: Optional. The cube noise level.
      nsigma: Optional. Noise level to filter data.
      edge: Optional. Border channels to ignore.
      log: Optional. Logging function.
    """
    # Load cube
    imgmax = np.nanmax(cube.filled_data[edge:-edge,:,:], axis=0)

    # Replace values below rms
    if rms is not None:
        log('Replacing values below %f by zero', nsigma * rms)
        imgmax[imgmax < nsigma * rms] = 0.

    return fits.PrimaryHDU(imgmax.value,
                           header=cube.wcs.sub(['longitude',
                                                'latitude']).to_header())

def find_peak(cube: Optional[u.Quantity] = None,
              image: Optional[fits.PrimaryHDU] = None,
              rms: Optional[u.Quantity] = None,
              collapse_func: Callable = max_collapse,
              diff: Optional[Sequence[npt.ArrayLike]] = None,
              log: Callable = print,
              **kwargs) -> Tuple[npt.ArrayLike, SkyCoord]:
    """Find an emission peak.

    If `image` is given, then the position of the maximum is given. Else,
    if `cube` is given, then it is collapsed along the spectral axis using the
    `collapse_func`, and the location of the maximum is returned. Otherwise
    `ValueError` is raised.

    Args:
      cube: Optional. Spectral cube.
      image: Optional. Collapsed image.
      rms: Optional. Cube noise level.
      collapse_func: Optional. Collapse function.
      diff: Optional. Differentials of the image along each axis.
      log: Optional. Logging function.
      kwargs: Additional arguments to `collapse_func`.

    Returns
      The collapsed image, the sky coordinate of the peak.
    """
    # Collapsed image
    if image is not None:
        log('Looking peak in input image')
        collapsed = image
        wcs = WCS(image.header, naxis=2)
    elif cube is not None:
        log('Looking peak in collapsed cube')
        collapsed = collapse_func(cube, rms=rms, log=log, **kwargs)
        wcs = cube.wcs.sub(['longitude', 'latitude'])
    else:
        raise ValueError('Cannot find spectrum position')

    # Find peak
    if diff is None:
        ymax, xmax = np.unravel_index(np.nanargmax(collapsed.data),
                                      collapsed.data.shape)
    else:
        # Search for peaks
        xmax = np.diff((diff[1] > 0).view(np.int8), axis=1)
        ymax = np.diff((diff[0] > 0).view(np.int8), axis=0)
        indxy, indxx = np.where(xmax == -1)
        indyy, indyx = np.where(ymax == -1)
        indxx = indxx + 1
        indyy = indyy + 1

        # Select the one with the highest value
        ind = np.argsort(collapsed.data[indxy, indxx])[::-1]
        indxy = indxy[ind]
        indxx = indxx[ind]
        for p in zip(indxx, indxy):
            if p in zip(indyx, indyy):
                if (rms is not None and
                    collapsed.data[p[1], p[0]] <= rms.to(cube.unit).value):
                    continue
                xmax, ymax = p
                break

    # Convert to coordinate
    coord = SkyCoord.from_pixel(xmax, ymax, wcs)

    return collapsed, coord

def get_common_position(images: Sequence['pathlib.Path'],
                        method: str,
                        save_collapsed: bool = False,
                        resume: bool = False,
                        log: Callable = print) -> Tuple[int]:
    """Find the position of peak emission common to all input cubes.

    Args:
      images: List of image files.
      method: Method to collapse the images (max, sum).
      save_collapsed: Optional. Save collapsed image.
      resume: Optional. Resume calculation.
      log: Optional. Logging function.

    Returns:
      A tuple with (x, y) positions.
    """
    # Find peaks
    positions = []
    for image in images:
        # Potential collapsed name
        log('-' * 15)
        imagename = image.with_suffix(f'.{method}.fits')

        # Restore
        if resume and imagename.exists():
            log(f'Loading collapsed image: {imagename}')
            collapsed = fits.open(imagename)[0]
            _, position = find_peak(image=collapsed)
            positions.append(position)
            continue

        # Find peak
        log(f'Finding position in cube: {image}')
        if method == 'sum':
            collapse_func = sum_collapse
        elif method == 'max':
            collapse_func = max_collapse
        else:
            raise NotImplementedError(f'Collapse method: {method}')
        cube = open_cube(image)

        # Find peak
        collapsed, position = find_peak(cube=cube,
                                        collapse_func=collapse_func,
                                        log=log)
        positions.append(position)

        # Save collapsed
        if save_collapsed:
            log(f'Saving collapsed image: {imagename}')
            collapsed.writeto(imagename, overwrite=True)

    # Combine peaks
    log(f'Individual peaks: {positions}')
    position = combine_positions(np.array(positions))
    log(f'Combined position: {position}')

    return position

def get_spectrum(spec_file: Optional['pathlib.Path'] = None,
                 cube_file: Optional['pathlib.Path'] = None,
                 position: Optional[SkyCoord] = None,
                 beam_avg: bool = False,
                 beam: Optional[u.Quantity] = None,
                 resume: bool = False,
                 log: Callable = print) -> Tuple[npt.ArrayLike, Tuple[int]]:
    """Load spectrum.

    At least one `(cube_file, position)` or `spec_file` must be specified. If
    `resume=True`, the file `spec_file` has priority, else the spectrum is
    recalculated from `cube_file` (if present).
    
    Args:
      spec_file: Optional. Spectrum file.
      cube_file: Optional. Cube file name.
      position: Optional. Position where to extract the spectrum from.
      beam_avg: Optional. Use a beam average?
      beam: Optional. Beam of the data.
      resume: Optional. Continure from where it left?
      log: Optional. Logging function.

    Returns:
      An array with the frequency axis
      An array with the intensity axis.
    """
    # Generate a spec_file name
    use_cube = cube_file is not None and position is not None
    if spec_file is None and use_cube:
        suffix = f'.ra{position.ra.deg:.4f}_dec{position.dec.deg:.4f}.spec.dat'
        spec_file = cube_file.with_suffix(suffix)

    # Load spec
    use_spec = spec_file is not None and spec_file.is_file()
    if use_spec and resume:
        # Load from file
        freq, spectrum = np.loadtxt(spec_file, usecols=(0, 1), unpack=True,
                                    dtype=float)
        log(f'Spectrum size: {spectrum.size}')
        freq = freq * u.GHz
        spectrum = spectrum * u.Jy/u.beam
    elif use_cube:
        # Load cube
        #cube = fits.open(cube_file)[0]
        cube = open_cube(cube_file)
        #header = cube.header

        # Remove dummy axes
        #cube = np.squeeze(cube.data) * u.Unit(header['BUNIT'])
        cube = cube.to(u.Jy/u.beam)
        cube = cube.with_spectral_unit(u.GHz)
        wcs = cube.wcs.sub(['longitude', 'latitude'])
        xpix, ypix = tuple(map(int, position.to_pixel(wcs)))
        freq = cube.spectral_axis
        log('Cube shape: %s', cube.shape)

        # Get spectrum at position
        if beam_avg:
            # Beam size
            log('Averaging over beam')
            pixsize = np.sqrt(wcs.proj_plane_pixel_area())
            if beam is None:
                if hasattr(cube, 'beam'):
                    beam = cube.beam
                else:
                    beam = cube.beams.common_beam()
            radius = np.sqrt(beam.sr / (2 * np.pi))
            radius = radius / pixsize.to(radius.unit)
            log('Beam avg radius (sigma) = %f pix', radius)

            # Filter data
            y_cube, x_cube = np.indices(cube.shape[-2:])
            dist = np.sqrt((x_cube - xpix)**2. +
                           (y_cube - ypix)**2.)
            mask = dist > radius
            masked = np.ma.array(cube.unmasked_data[:],
                                 mask=np.tile(mask, (cube.shape[0], 1)))
            spectrum = np.ma.sum(masked, axis=(1, 2)) / np.sum(~mask)
        else:
            log('Using single pixel spectrum')
            spectrum = cube.unmasked_data[:, ypix, xpix]
        log(f'Number of channels: {spectrum.size}')

        # Save to file
        with spec_file.open('w') as out:
            lines = [f'{nu.value:.10f}\t{fnu.value}'
                     for nu, fnu in zip(freq, spectrum)]
            out.write('\n'.join(lines))

    else:
        raise ValueError('Cannot find a valid spectrum')

    return freq, spectrum

def pb_crop(imagename: 'pathlib.Path',
            pbmap: 'pathlib.Path',
            level: float) -> 'pathlib.Path':
    """Crop image based on PB limit.

    Args:
      imagename: Image file name.
      pbmap: PB map file name.
      level: PB map level for cutout.

    Returns:
      The output file name.
    """
    # Determine type
    suffix = imagename.suffix
    if suffix == '.fits':
        dtype = 'fits'
    else:
        dtype = 'casa'

    # Load images
    cube = open_cube(imagename, import_format=dtype)
    pbmap = open_cube(pbmap, import_format=dtype)

    # Get subcube
    cube_slice = cube.subcube_slices_from_mask(pbmap > level,
                                               spatial_only=True)
    cube = cube[cube_slice]

    # Save
    outfile = imagename.with_suffix('.crop' + suffix)
    cube.write(outfile)

    return outfile

