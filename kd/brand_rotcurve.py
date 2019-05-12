#!/usr/bin/env python
"""
brand_rotcurve.py

Utilities involving the Brand rotation curve.

2017-05-24 Trey V. Wenger
"""

import numpy as np

from kd import kd_utils

#
# Brand rotation curve parameters
#
__a1 = 1.0074
__a2 = 0.0382
__a3 = 0.00698
__R0 = 8.5 # kpc
__theta0 = 220.0 # km/s

def calc_theta(R,a1=__a1,a2=__a2,a3=__a3,R0=__R0):
    """
    Return circular orbit speed theta at given Galactocentric radius
    R.

    Parameters:
      R : scalar or 1-D array
          Galactocentric radius (kpc)
      a1,a2,a3 : scalars (optional)
                 Brand rotation curve parameters
      R0 : scalar (optional)
           Solar Galactocentric radius (kpc)

    Returns: theta
      theta : scalar or 1-D array
              circular orbit speed at R (km/s)
    """
    theta = __theta0 * (a1*(R/R0)**a2 + a3)
    return theta

def calc_vlsr(glong, dist, resample=False):
    """
    Return the LSR velocity at a given Galactic longitude and
    line-of-sight distance.
    Resampling is not available for this rotation curve

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as dist.
      dist : scalar or 1-D array
             line-of-sight distance (kpc). If it is an array, it must
             have the same size as glong.
      resample : bool (optional)
                 if True, resample rotation curve parameters within
                 uncertainties (NOT AVAILABLE)

    Returns: vlsr, params
      vlsr : scalar or 1-D array
             LSR velocity (km/s). If glong and dist are scalars, it
             is a scalar. Otherwise it has shape (dist.size).

      params : dict of scalars
        parameters used to calculate vlsr (useful if resample is True)
        params["R0"] : R0 used in calculation
        params["a1"] : a1 used in calculation
        params["a2"] : a2 used in calculation
        params["a3"] : a3 used in calculation

    Raises:
      ValueError : if glong or dist are not 1-D; or
                   if glong and dist are arrays and not the same size
                   if resample is True
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, dist_inp = np.atleast_1d(glong, dist)
    # check shape of inputs
    if glong_inp.ndim != 1 or dist_inp.ndim != 1:
        raise ValueError("glong and dist must be 1-D")
    if glong_inp.size != 1 and glong_inp.size != dist_inp.size:
        raise ValueError("glong and dist must have same size")
    if resample:
        raise ValueError("Re-sampling not available for Brand rotcurve")
    a1 = __a1
    a2 = __a2
    a3 = __a3
    R0 = __R0
    params = {"R0":R0,"a1":a1,"a2":a2,"a3":a3}
    #
    # Convert distance to Galactocentric radius, catch places where
    # R = 0.
    #
    Rgal = kd_utils.calc_Rgal(glong_inp,dist_inp,R0=R0)
    Rgal = np.atleast_1d(Rgal)
    Rgal[Rgal < 1.e-6] = 1.e-6
    #
    # Brand rotation curve circular velocity
    #
    theta = calc_theta(Rgal,a1=a1,a2=a2,a3=a3,R0=R0)
    #
    # Now take circular velocity and convert to LSR velocity
    #
    vlsr = R0 * np.sin(np.deg2rad(glong_inp))
    vlsr = vlsr * ((theta/Rgal) - __theta0/R0)
    #
    # Convert back to scalar if necessary
    #
    if dist_inp.size == 1:
        return vlsr[0],params
    else:
        return vlsr,params
