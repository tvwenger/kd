#!/usr/bin/env python
"""
parallax.py

Utility to calculate parallax distances the traditional way.

2019-01-19 Trey V. Wenger
"""

import numpy as np

from kd import kd_utils

def parallax(glong, plx, dist_max=30., R0=8.34):
    """
    Compute parallax distance and Galactocentric radius for a given
    Galactic longitude and parallax.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as plx.
      plx : scalar or 1-D array
            Parallax (milli-arcseconds). If it is an array, it must
            have the same size as glong.
      dist_max : scalar (optional)
                 The maximum parallax distance to compute (kpc)
      R0 : scalar (optional)
           Solar Galactocentric radius (kpc)

    Returns: output
      output["Rgal"] : scalar or 1-D array
                       Galactocentric radius (kpc).
      output["distance"] : scalar or 1-D array
                           Parallax distance (kpc).
      If glong and plx are scalars, each of these is a scalar.
      Otherwise they have shape (glong.size).

    Raises:
      ValueError : if glong and plx are not 1-D; or
                   if glong and plx are arrays and not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, plx_inp = np.atleast_1d(glong, plx)
    # check shape of inputs
    if glong_inp.ndim != 1 or plx_inp.ndim != 1:
        raise ValueError("glong and plx must be 1-D")
    if glong_inp.size != plx_inp.size:
        raise ValueError("glong and plx must have same size")
    # ensure range [0,360) degrees
    fix_glong = glong_inp % 360.
    #
    # Compute distances
    #
    distance = np.array(1./plx_inp) # kpc
    # remove large distances
    distance[distance > dist_max] = np.nan
    #
    # Compute Galactocentric radius
    #
    Rgal = np.array([kd_utils.calc_Rgal(l,d,R0=R0)
                     for l,d in zip(fix_glong, distance)])
    #
    # Convert back to scalars if necessary
    #
    if len(fix_glong) == 1:
        return {"Rgal":Rgal[0], "distance":distance[0]}
    else:
        return {"Rgal":Rgal, "distance":distance}

