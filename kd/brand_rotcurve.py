#!/usr/bin/env python
"""
brand_rotcurve.py

Utilities involving the Brand & Blitz (1993) rotation curve.

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

2017-05-24 Trey V. Wenger
2020-02-19 Trey V. Wenger updates for v2.0
2021-11-09 Trey V. Wenger fix bug in calc_vlsr
"""

import numpy as np
from kd import kd_utils

#
# Brand & Blitz (1993) nominal rotation curve parameters
#
__a1 = 1.0074
__a2 = 0.0382
__a3 = 0.00698
__R0 = 8.5  # kpc
__theta0 = 220.0  # km/s


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
    params = {"a1": __a1, "a2": __a2, "a3": __a3, "R0": __R0, "theta0": __theta0}
    return params


def resample_params(size=None):
    """
    This rotation curve does not support resampling. This function
    raises an error.

    Parameters:
      size : integer
        Unused

    Returns: Nothing
    """
    raise ValueError("brand_rotcurve.py does not support resampling")


def calc_theta(R, a1=__a1, a2=__a2, a3=__a3, R0=__R0, theta0=__theta0):
    """
    Return circular orbit speed at a given Galactocentric radius.

    Parameters:
      R :: scalar or array of scalars
        Galactocentric radius (kpc)

      a1, a2, a3 :: scalars (optional)
        Brand rotation curve parameters

      R0, theta0 :: scalars (optional)
        Solar Galactocentric radius (kpc) and circular orbit speed at
        R0 (km/s)

    Returns: theta
      theta :: scalar or array of scalars
        circular orbit speed at R (km/s)
    """
    input_scalar = np.isscalar(R)
    R = np.atleast_1d(R)
    theta = theta0 * (a1 * (R / R0) ** a2 + a3)
    if input_scalar:
        return theta[0]
    return theta


def calc_vlsr(glong, glat, dist, a1=__a1, a2=__a2, a3=__a3, R0=__R0, theta0=__theta0):
    """
    Return the LSR velocity at a given Galactic longitude and
    line-of-sight distance.

    Parameters:
      glong, glat :: scalar or array of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      a1, a2, a3 :: scalars (optional)
        Brand rotation curve parameters

      R0, theta0 :: scalars (optional)
        Solar Galactocentric radius (kpc) and circular orbit speed at
        R0 (km/s)

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
    theta = calc_theta(Rgal, a1=a1, a2=a2, a3=a3, R0=R0, theta0=theta0)
    #
    # Now take circular velocity and convert to LSR velocity
    #
    vlsr = R0 * np.sin(np.deg2rad(glong))
    vlsr = vlsr * ((theta / Rgal) - theta0 / R0)
    if input_scalar:
        return vlsr[0]
    return vlsr
