#!/usr/bin/env python
"""
reid14_rotcurve.py

Utilities involving the Universal Rotation Curve (Persic+1996) from
Reid+2014.

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

2017-04-12 Trey V. Wenger
2020-02-19 Trey V. Wenger updates for v2.0
"""

import numpy as np
from kd import kd_utils

#
# Reid+2014 rotation curve parameters and uncertainties
#
__a1 = 241.0  # km/s V(R_opt)
__a1_err = 8.0
__a2 = 0.90  # R_opt/ R0
__a2_err = 0.06
__a3 = 1.46  # 1.5*(L/L*)^0.2
__a3_err = 0.16
__R0 = 8.34  # kpc
__R0_err = 0.16


def nominal_params():
    """
    Return a dictionary containing the nominal rotation curve
    parameters.

    Parameters: Nothing

    Returns: params
      params :: dictionary
        params['a1'], etc. : scalar
          The nominal rotation curve parameter
    """
    params = {"a1": __a1, "a2": __a2, "a3": __a3, "R0": __R0}
    return params


def resample_params(size=None):
    """
    Resample the Reid+2014 rotation curve parameters within their
    uncertainties assuming Gaussian probabilities.

    Parameters:
      size :: integer
        The number of random samples to generate. If None, generate
        only one sample and return a scalar.

    Returns: params
      params :: dictionary
        params['a1'], etc. : scalar or array of scalars
                             The re-sampled parameters
    """
    params = {
        "a1": np.random.normal(loc=__a1, scale=__a1_err, size=size),
        "a2": np.random.normal(loc=__a2, scale=__a2_err, size=size),
        "a3": np.random.normal(loc=__a3, scale=__a3_err, size=size),
        "R0": np.random.normal(loc=__R0, scale=__R0_err, size=size),
    }
    return params


def calc_theta(R, a1=__a1, a2=__a2, a3=__a3, R0=__R0):
    """
    Return circular orbit speed at a given Galactocentric radius.

    Parameters:
      R :: scalar or array of scalars
        Galactocentric radius (kpc)

      a1, a2, a3 :: scalars (optional)
        Reid+2014 rotation curve parameters

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

    Returns: theta
      theta :: scalar or array of scalars
        circular orbit speed at R (km/s)
    """
    input_scalar = np.isscalar(R)
    R = np.atleast_1d(R)
    #
    # Equations 8, 9, 10, 11a, 11b in Persic+1996
    #
    x = R / (a2 * R0)
    LLstar = (a3 / 1.5) ** 5.0
    beta = 0.72 + 0.44 * np.log10(LLstar)
    # Disk component Vd^2 / V(R_opt)^2
    Vd2 = beta * 1.97 * x ** 1.22 / (x ** 2.0 + 0.78 ** 2.0) ** 1.43
    # Halo component Vh^2 / V(R_opt)^2
    Vh2 = (1.0 - beta) * (1.0 + a3 ** 2.0) * x ** 2.0 / (x ** 2.0 + a3 ** 2.0)
    #
    # Catch non-physical case where Vd2 + Vh2 < 0
    #
    Vtot = Vd2 + Vh2
    Vtot[Vtot < 0.0] = np.nan
    #
    # Circular velocity
    #
    theta = a1 * np.sqrt(Vtot)
    if input_scalar:
        return theta[0]
    return theta


def calc_vlsr(glong, glat, dist, a1=__a1, a2=__a2, a3=__a3, R0=__R0):
    """
    Return the LSR velocity at a given Galactic longitude and
    line-of-sight distance.

    Parameters:
      glong, glat :: scalar or array of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      a1, a2, a3 :: scalars (optional)
        Reid+2014 rotation curve parameters

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

    Returns: vlsr
      vlsr :: scalar or array of scalars
        LSR velocity (km/s).
    """
    input_scalar = np.isscalar(glong) and np.isscalar(glat) and np.isscalar(dist)
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    #
    # Convert distance to Galactocentric radius, catch small Rgal
    #
    Rgal = kd_utils.calc_Rgal(glong, glat, dist, R0=R0)
    Rgal[Rgal < 1.0e-6] = 1.0e-6
    #
    # Rotation curve circular velocity
    #
    theta = calc_theta(Rgal, a1=a1, a2=a2, a3=a3, R0=R0)
    theta0 = calc_theta(R0, a1=a1, a2=a2, a3=a3, R0=R0)
    #
    # Now take circular velocity and convert to LSR velocity
    #
    vlsr = R0 * np.sin(np.deg2rad(glong))
    vlsr = vlsr * ((theta / Rgal) - (theta0 / R0))
    if input_scalar:
        return vlsr[0]
    return vlsr
