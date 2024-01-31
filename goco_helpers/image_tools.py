"""Tools for working with cubes and 2D images."""
from typing import Optional, Callable, Tuple, Sequence

from astropy.io import fits
from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import numpy.typing as npt

def open_cube(imagename: 'pathlib.Path',
              import_format: str = 'casa') -> SpectralCube:
    """Read a cube."""
    image = SpectralCube.read(imagename, use_dask=False, format=import_format)
    image.allow_huge_operations = True
    image.use_dask_scheduler('threads', num_workers=12)

    return image

def combine_positions(positions: npt.ArrayLike,
                      method: str = 'mean') -> Tuple[int]:
    """Combine position using the requested method."""
    if method == 'mean':
        if len(positions) <= 2:
            return tuple(np.mean(positions, axis=0, dtype=int))
        mean = np.mean(positions, axis=0)
        dist = np.sqrt((positions[:,0] - mean[0])**2 +
                       (positions[:,1] - mean[1])**2)
        ind = np.nanargmax(dist)
        best_pos = np.delete(positions, ind, axis=0)
        best_pos = np.mean(best_pos, axis=0, dtype=int)
    else:
        raise NotImplementedError(f'Combine method: {method}')

    return tuple(best_pos)

def sum_collapse(cube: SpectralCube,
                 rms: Optional[u.Quantity] = None,
                 nsigma: float = 5.,
                 edge: int = 10,
                 log: Callable = print) -> npt.ArrayLike:
    """Collapse cube along the spectral axis.

    If `rms` is given, then all values over `nsigma*rms` are summed.

    Args:
      cube: spectral cube.
      rms: optional; the cube noise level.
      nsigma: optional; noise level to filter data.
      edge: optional; border channels to ignore.
      log: optional; logging function.
    """
    if rms is not None:
        log(f'Summing all values over: {nsigma * rms}')
        masked = cube.with_mask(cube > nsigma * rms)
        masked = masked.with_fill_value(0.)
        imgsum = np.sum(masked.filled_data[edge:-edge, :, :], axis=0)
    else:
        log('Summing along spectral axis')
        imgsum = np.sum(cube.filled_data[edge:-edge, :, :], axis=0)

    return imgsum

def max_collapse(cube: SpectralCube,
                 rms: Optional[u.Quantity] = None,
                 nsigma: float = 1.,
                 edge: int = 10,
                 log: Callable = print) -> npt.ArrayLike:
    """Collapse cube along the spectral axis using `max` function.

    If `rms` is given, then all values below `nsigma*rms` are set to zero.

    Args:
      cubename: cube filename.
      rms: optional; the cube noise level.
      nsigma: optional; noise level to filter data.
      edge: optional; border channels to ignore.
      log: optional; logging function.
    """
    # Load cube
    imgmax = np.nanmax(cube.filled_data[edge:-edge,:,:], axis=0)

    # Replace values below rms
    if rms is not None:
        log('Replacing values below %f by zero', nsigma * rms)
        imgmax[imgmax < nsigma * rms] = 0.

    return imgmax

def find_peak(cube: Optional[u.Quantity] = None,
              image: Optional[npt.ArrayLike] = None,
              rms: Optional[u.Quantity] = None,
              collapse_func: Callable = max_collapse,
              diff: Optional[Sequence[npt.ArrayLike]] = None,
              log: Callable = print,
              **kwargs) -> Tuple[npt.ArrayLike, int, int]:
    """Find an emission peak.

    If `image` is given, then the position of the maximum is given. Else,
    if `cube` is given, then it is collapsed along the spectral axis using the
    `collapse_func`, and the location of the maximum is returned. Otherwise
    `ValueError` is raised.

    Args:
      cube: spectral cube.
      image: 2-D image.
      rms: optional; cube noise level.
      collapse_func: optional; collapse function.
      diff: optional; differentials of the image along each axis.
      log: optional; logging function.
      kwargs: additional arguments to `collapse_func`
    """
    # Collapsed image
    if image is not None:
        log('Looking peak in input image')
        collapsed = image
    elif cube is not None:
        log('Looking peak in collapsed cube')
        collapsed = collapse_func(cube.value, rms=rms.to(cube.unit).value,
                                  log=log, **kwargs)

    # Find peak
    if diff is None:
        ymax, xmax = np.unravel_index(np.nanargmax(collapsed), collapsed.shape)
    else:
        # Search for peaks
        xmax = np.diff((diff[1] > 0).view(np.int8), axis=1)
        ymax = np.diff((diff[0] > 0).view(np.int8), axis=0)
        indxy, indxx = np.where(xmax == -1)
        indyy, indyx = np.where(ymax == -1)
        indxx = indxx + 1
        indyy = indyy + 1

        # Select the one with the highest value
        ind = np.argsort(collapsed[indxy, indxx])[::-1]
        indxy = indxy[ind]
        indxx = indxx[ind]
        for p in zip(indxx, indxy):
            if p in zip(indyx, indyy):
                if (rms is not None and
                    collapsed[p[1], p[0]] <= rms.to(cube.unit).value):
                    continue
                xmax, ymax = p
                break

    return collapsed, xmax, ymax

def get_common_position(images: Sequence['pathlib.Path'],
                        method: str,
                        save_collapsed: bool = False,
                        resume: bool = False,
                        log: Callable = print) -> Tuple[int]:
    """Find the position of peak emission common to all input cubes.

    Args:
      images: list of image files.
      method: method to collapse the images (max, sum).
      save_collapsed: optional; save collapsed image.
      resume: optional; resume calculation.
      log: optional; logging function.

    Returns:
      A tuple with (x, y) positions.
    """
    # Find peaks
    positions = []
    for image in images:
        # Potential collapsed name
        imagename = image.with_suffix(f'.{method}.fits')

        # Restore
        if resume and imagename.exists():
            collapsed = fits.open(imagename)
            _, *position = find_peak(image=collapsed.data)
            positions.append(position)
            continue

        # Find peak
        log('Finding position in cube: %s', image)
        if method == 'sum':
            collapse_func = sum_collapse
        elif method == 'max':
            collapse_func = max_collapse
        else:
            raise NotImplementedError(f'Collapse method: {method}')
        cube = open_cube(image)

        # Find peak
        collapsed, *position = find_peak(cube=cube,
                                         collapse_func=collapse_func,
                                         log=log)
        positions.append(position)

        # Save collapsed
        if save_collapsed:
            wcs = cube.wcs.sub(['logitude', 'latitude'])
            hdu = fits.PrimaryHDU(collapsed, header=wcs.to_header())
            hdu.writeto(imagename, overwrite=True)

    # Combine peaks
    log(f'Individual peaks: {positions}')
    position = combine_positions(np.array(positions))
    log(f'Combined position: {position}')

    return position

def get_spectrum(spec_file: Optional['pathlib.Path'] = None,
                 cube_file: Optional['pathlib.Path'] = None,
                 position: Tuple[int] = None,
                 beam_avg: bool = False,
                 beam: Optional[u.Quantity] = None,
                 resume: bool = False,
                 log: Callable = print) -> Tuple[npt.ArrayLike, Tuple[int]]:
    """Load spectrum.

    At least one `(cube_file, position)` or `spec_file` must be specified. If
    `resume=True`, the file `spec_file` has priority, else the spectrum is
    recalculated from `cube_file` (if present).
    
    Args:
      spec_file: spectrum file.
      cube_file: cube file name.
      position: optional; position where to extract the spectrum from.
      beam_avg: optional; use a beam average?
      beam: optional; beam of the data.
      resume: optional; 
      log: optional; logging function.

    Returns:
      An array with the frequency axis
      An array with the intensity axis.
    """
    # Generate a spec_file name
    use_cube = cube_file is not None and position is not None
    if spec_file is None and use_cube:
        suffix = f'.x{position[0]}_y{position[1]}.spec.dat'
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
        freq = cube.spectral_axis
        log('Cube shape: %s', cube.shape)

        # Get spectrum at position
        if beam_avg:
            # Beam size
            log('Averaging over beam')
            wcs = cube.wcs.sub(['longitude', 'latitude'])
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
            dist = np.sqrt((x_cube - position[0])**2. +
                           (y_cube - position[1])**2.)
            mask = dist > radius
            masked = np.ma.array(cube.unmasked_data[:],
                                 mask=np.tile(mask, (cube.shape[0], 1)))
            spectrum = np.ma.sum(masked, axis=(1, 2)) / np.sum(~mask)
        else:
            log('Using single pixel spectrum')
            spectrum = cube.unmasked_data[:, position[1], position[0]]
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
      imagename: image file name.
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
    pbmap = open_cube(imagename, import_format=dtype)

    # Get subcube
    cube_slice = cube.subcube_slices_from_mask(pbmap > level,
                                               spatial_only=True)
    cube = cube[cube_slice]

    # Save
    outfile = imagename.with_suffix('.crop' + suffix)
    cube.write(outfile)

    return outfile

