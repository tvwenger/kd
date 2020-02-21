#!/usr/bin/env python
"""
parallax.py

Utility to calculate parallax distances the traditional way.

Copyright(C) 2017-2020 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

2019-01-19 Trey V. Wenger
2020-02-19 Trey V. Wenger updates for v2.0
"""

import numpy as np
from kd import kd_utils

# Solar Galactocentric radius and error from Reid+2019
__R0 = 8.15 # kpc
__R0_err = 0.15 # kpc

def parallax(glong, plx, plx_err=None, dist_max=30., R0=__R0,
             R0_err=__R0_err, resample=False, size=1):
    """
    Compute parallax distance and Galactocentric radius for a given
    Galactic longitude and parallax.

    Parameters:
      glong :: scalar or array of scalars
        Galactic longitude (deg).

      plx :: scalar or array of scalars
        Parallax (milli-arcseconds).

      plx_err :: scalar or 1-D (optional)
        Parallax uncertainty in milli-arcseconds. If it is an array,
        it must have the same size as plx. Otherwise, this scalar
        uncertainty is applied to all plxs.

      dist_max :: scalar (optional)
        The maximum parallax distance to compute (kpc)

      R0, R0_err :: scalar (optional)
        Solar Galactocentric radius and uncertainty (kpc)

    Returns: output
      output["Rgal"] :: scalar or array of scalars
        Galactocentric radius (kpc).

      output["distance"] :: scalar or array of scalars
        Parallax distance (kpc).
    """
    #
    # check inputs
    #
    # check shape of inputs
    glong, plx = np.atleast_1d(glong, plx)
    inp_shape = glong.shape
    glong = glong.flatten()
    plx = plx.flatten()
    if glong.shape != plx.shape:
        raise ValueError("glong and plx must have same size")
    if (plx_err is not None and not np.isscalar(plx_err)):
        plx_err = plx_err.flatten()
        if plx_err.shape != plx.shape:
            raise ValueError("plx_err must be scalar or have same shape as plx")
    #
    # Default velo_err to 0, sample size to 1
    #
    elif plx_err is None:
        plx_err = 0.
    if not resample:
        size = 1
    if size < 1:
        raise ValueError("size must be >= 1")
    # ensure range [0,360) degrees
    glong = glong % 360.
    #
    # Storage for results
    #
    distance_samples = np.ones((len(glong), size), dtype=float)*np.nan
    Rgal_samples = np.ones((len(glong), size), dtype=float)*np.nan
    #
    # Get nominal parameters
    #
    R0_sample = R0
    plx_sample = np.copy(plx)
    for snum in range(size):
        #
        # Resample parallax and R0
        #
        if resample:
            plx_sample = np.random.normal(loc=plx, scale=plx_err)
            R0_sample = np.random.normal(loc=R0, scale=R0_err)
        #
        # Compute distances from parallax, catch large distances
        #
        distance = 1./plx_sample # kpc
        distance[distance > dist_max] = np.nan
        #
        # Compute Galactocentric radius
        #
        Rgal = kd_utils.calc_Rgal(glong, distance, R0=R0_sample)
        #
        # Save
        #
        distance_samples[:, snum] = distance
        Rgal_samples[:, snum] = Rgal
    #
    # Convert back to scalars if necessary
    #
    output = {"Rgal": Rgal_samples, "distance": distance_samples}
    if size == 1:
        for key in output:
            output[key] = np.squeeze(output[key], axis=-1)
    if glong.size == 1:
        for key in output:
            output[key] = output[key][0]
    else:
        for key in output:
            if size == 1:
                output[key] = output[key].reshape(inp_shape)
            else:
                output[key] = output[key].reshape(inp_shape+(size,))
    return output
