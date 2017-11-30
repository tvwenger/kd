#!/usr/bin/env python
"""
reid_rotcurve.py

Utilities involving the Universal Rotation Curve (Persic+1996) from
Reid+2014.

2017-04-12 Trey V. Wenger
"""

import numpy as np

from kd import kd_utils

#
# Reid+2014 rotation curve parameters
#
__a1 = 241. # km/s V(R_opt)
__a1_err = 8.
__a2 = 0.90 # R_opt/ R0
__a2_err = 0.06
__a3 = 1.46 # 1.5*(L/L*)^0.2
__a3_err = 0.16
__R0 = 8.34 # kpc
__R0_err = 0.16

def calc_theta(R_inp,a1=__a1,a2=__a2,a3=__a3,R0=__R0,resample=False):
    """
    Return circular orbit speed theta at given Galactocentric radius
    R.

    Parameters:
      R : scalar or 1-D array
          Galactocentric radius (kpc)
      a1,a2,a3 : scalars (optional)
                 Reid+2014 rotation curve parameters
      R0 : scalar (optional)
           Solar Galactocentric radius (kpc)

    Returns: theta
      theta : scalar or 1-D array
              circular orbit speed at R (km/s)
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    R = np.atleast_1d(R_inp)
    #
    # Resample rotation curve parameters if necessary
    #
    if resample:
        # resample fit parameters within uncertainty
        a1 = np.random.normal(loc=__a1,scale=__a1_err)
        a2 = np.random.normal(loc=__a2,scale=__a2_err)
        a3 = np.random.normal(loc=__a3,scale=__a3_err)
        R0 = np.random.normal(loc=__R0,scale=__R0_err)
    #
    # Equations 8, 9, 10, 11a, 11b in Persic+1996
    #
    x = R/(a2 * R0)
    LLstar = (a3/1.5)**5.
    beta = 0.72 + 0.44*np.log10(LLstar)
    # Disk component Vd^2 / V(R_opt)^2
    Vd2 = beta * 1.97 * x**1.22 / (x**2. + 0.78**2.)**1.43
    # Halo component Vh^2 / V(R_opt)^2
    Vh2 = (1.-beta)*(1.+a3**2.)*x**2./(x**2. + a3**2.)
    #
    # Catch non-physical case where Vd2 + Vh2 < 0
    #
    Vtot = Vd2 + Vh2
    Vtot[Vtot < 0.] = np.nan
    #
    # Circular velocity
    #
    theta = a1 * np.sqrt(Vtot)
    return theta

def calc_vlsr(glong, dist, resample=False):
    """
    Return the LSR velocity at a given Galactic longitude and
    line-of-sight distance.
    If requested, resample rotation curve parameters and R0 within
    uncertainties assuming Gaussian errors.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as dist.
      dist : scalar or 1-D array
             line-of-sight distance (kpc). If it is an array, it must
             have the same size as glong or glong must be a scalar.
      resample : bool (optional)
                 if True, resample rotation curve parameters within
                 uncertainties

    Returns: vlsr, params
      vlsr : scalar or 1-D array
             LSR velocity (km/s). If dist is a scalar, it
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
        raise ValueError("glong and dist must have same size, "
                         "or glong must be a scalar")
    #
    # Resample rotation curve parameters if necessary
    #
    if resample:
        # resample fit parameters within uncertainty
        a1 = np.random.normal(loc=__a1,scale=__a1_err)
        a2 = np.random.normal(loc=__a2,scale=__a2_err)
        a3 = np.random.normal(loc=__a3,scale=__a3_err)
        R0 = np.random.normal(loc=__R0,scale=__R0_err)
    else:
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
    # Reid rotation curve circular velocity
    #
    theta = calc_theta(Rgal,a1=a1,a2=a2,a3=a3,R0=R0)
    #
    # Now take circular velocity and convert to LSR velocity
    #
    vlsr = R0 * np.sin(np.deg2rad(glong_inp))
    vlsr = vlsr * ((theta/Rgal) - a1/R0)
    #
    # Convert back to scalar if necessary
    #
    if dist_inp.size == 1:
        return vlsr[0],params
    else:
        return vlsr,params
