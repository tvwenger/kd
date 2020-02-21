#!/usr/bin/env python
"""
reid19_rotcurve.py

Utilities involving the Universal Rotation Curve (Persic+1996) from
Reid+2019, with re-done analysis by Wenger+2020.

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

2020-02-19 Trey V. Wenger new in V2.0
"""

import os
import pickle
import numpy as np
from kd import kd_utils

#
# Reid+2019 A5 rotation curve parameters
#
__a2 = 0.96
__a3 = 1.62
__R0 = 8.15

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
    params = {
        'a2': __a2, 'a3': __a3, 'R0': __R0}
    return params

def resample_params(size=None):
    """
    Resample the Reid+2019 rotation curve parameters within their
    uncertainties using the Wenger+2020 kernel density estimator
    to include parameter covariances.

    Parameters:
      size :: integer
        The number of random samples to generate. If None, generate
        only one sample and return a scalar.

    Returns: params
      params :: dictionary
        params['a1'], etc. : scalar or array of scalars
                             The re-sampled parameters
    """
    kdefile = os.path.join(
        os.path.dirname(__file__), 'reid19_params.pkl')
    with open(kdefile, 'rb') as f:
        kde = pickle.load(f)
    if size is None:
        samples = kde['full'].resample(1)
        params = {
            'a2': samples[6][0], 'a3': samples[7][0],
            'R0': samples[0][0]}
    else:
        samples = kde['full'].resample(size)
        params = {
            'a2': samples[6], 'a3': samples[7],
            'R0': samples[0]}
    return params

def calc_theta(R, a2=__a2, a3=__a3, R0=__R0):
    """
    Return circular orbit speed at a given Galactocentric radius.

    Parameters:
      R :: scalar or array of scalars
        Galactocentric radius (kpc)

      a2, a3 :: scalars (optional)
        Reid+2019 rotation curve parameters

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

    Returns: theta
      theta :: scalar or array of scalars
        circular orbit speed at R (km/s)
    """
    #
    # Re-production of Reid+2019 FORTRAN code
    #
    rho = R/(a2 * R0)
    lam = (a3/1.5)**5.
    loglam = np.log10(lam)
    term1 = 200. * lam**0.41
    term2 = np.sqrt(0.8 + 0.49*loglam +
                    0.75*np.exp(-0.4*lam)/(0.47 + 2.25*lam**0.4))
    term3 = (0.72 + 0.44*loglam) * 1.97 * rho**1.22 / (rho**2. + 0.61)**1.43
    term4 = 1.6 * np.exp(-0.4*lam) * rho**2. / (rho**2. + 2.25*lam**0.4)
    #
    # Catch non-physical case where term3 + term4 < 0
    #
    term = term3 + term4
    if np.isscalar(term) and term < 0.:
        term = np.nan
    elif not np.isscalar(term):
        term[term < 0.] = np.nan
    #
    # Circular velocity
    #
    theta = term1/term2 * np.sqrt(term)
    return theta

def calc_vlsr(glong, dist, a2=__a2, a3=__a3, R0=__R0):
    """
    Return the LSR velocity at a given Galactic longitude and
    line-of-sight distance.

    Parameters:
      glong :: scalar or array of scalars
        Galactic longitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      a2, a3 :: scalars (optional)
        Reid+2019 rotation curve parameters

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

    Returns: vlsr
      vlsr :: scalar or array of scalars
        LSR velocity (km/s).
    """
    #
    # Convert distance to Galactocentric radius, catch small Rgal
    #
    Rgal = kd_utils.calc_Rgal(glong, dist, R0=R0)
    if np.isscalar(Rgal) and Rgal < 1.e-6:
        Rgal = 1.e-6
    elif not np.isscalar(Rgal):
        Rgal[Rgal < 1.e-6] = 1.e-6
    #
    # Rotation curve circular velocity
    #
    theta = calc_theta(
        Rgal, a2=a2, a3=a3, R0=R0)
    theta0 = calc_theta(R0, a2=a2, a3=a3, R0=R0)
    #
    # Now take circular velocity and convert to LSR velocity
    #
    vlsr = R0 * np.sin(np.deg2rad(glong))
    vlsr = vlsr * ((theta/Rgal) - (theta0/R0))
    return vlsr
